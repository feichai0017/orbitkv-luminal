use anyhow::{Context, Result};
use luminal::prelude::*;

use crate::pt2_schema::*;
use crate::pt2_util::*;

use super::Translator;

/// Broadcast-add an additive offset (causal mask, attention bias) onto the
/// scores. Handles dtype promotion and rank extension: a `(S_q, S_k)` offset
/// broadcasts across any batch/head prefix dims of `scores`.
fn add_offset(scores: GraphTensor, offset: GraphTensor) -> GraphTensor {
    let (s, o) = ensure_same_dtype(scores, offset);
    let (s, o) = broadcast_binary(s, o);
    s + o
}

impl<'a> Translator<'a> {
    /// Translate all `scaled_dot_product_attention` ATen variants (unified,
    /// efficient, flash, flash_for_cpu, cudnn) into
    /// `softmax((Q@K^T)*scale + causal_mask + attn_bias) @ V`. Args resolve
    /// by name so one body serves every variant; only tuple slot 0 (the
    /// attention output) is bound — logsumexp and later slots are
    /// inference-time dead ends, left unbound by design.
    ///
    /// Torch-kernel parity notes: bf16/f16 run the score chain in F32, probs
    /// return to the value dtype for PV — mirroring torch's kernels, which
    /// compute in `opmath_type<scalar_t>` (= f32 for bf16/f16; see pytorch
    /// `aten/src/ATen/native/cpu/FlashAttentionKernel.cpp`, `accum_t`, and
    /// the FlashAttention-2 paper, arXiv:2307.08691 §3); `is_causal` is a
    /// top-left iota mask
    /// (symbolic-seq safe); bool masks are keep-masks and fully-masked query
    /// rows output zeros; grouped K/V heads are repeat-interleaved (GQA);
    /// Q/K head_dim and K/V seq SymInts are unified per the op contract.
    pub(crate) fn translate_sdpa(&mut self, node: &Node) -> Result<()> {
        let query = self.get_input_tensor(node, 0)?;
        let mut key = self.get_input_tensor(node, 1)?;
        let mut value = self.get_input_tensor(node, 2)?;

        let q_ndim = query.shape.len();
        anyhow::ensure!(
            q_ndim >= 2,
            "SDPA: query must have at least 2 dims (got {q_ndim})"
        );

        // Q/K share head_dim and K/V share seq by op contract, but dynamo
        // gives each placeholder its own SymInt — unify so matmul
        // dim-equality checks hold.
        if key.shape.len() == q_ndim && value.shape.len() == q_ndim {
            key.shape.dims[q_ndim - 1] = query.shape.dims[q_ndim - 1];
            value.shape.dims[q_ndim - 2] = key.shape.dims[q_ndim - 2];
        }

        let arg_by_name =
            |name: &str| -> Option<&NodeInput> { node.inputs.iter().find(|i| i.name == name) };
        let tensor_arg = |name: &str| -> Option<GraphTensor> {
            arg_by_name(name)
                .and_then(|i| i.arg.as_tensor_name())
                .and_then(|n| self.get_tensor(n).ok())
        };
        let float_arg =
            |name: &str| -> Option<f64> { arg_by_name(name).and_then(|i| i.arg.as_float()) };
        let bool_arg =
            |name: &str| -> Option<bool> { arg_by_name(name).and_then(|i| i.arg.as_bool()) };

        // attn_bias (Efficient/Cudnn/Unified) or attn_mask (FlashForCpu/Unified).
        let additive = tensor_arg("attn_bias").or_else(|| tensor_arg("attn_mask"));
        let is_causal = bool_arg("is_causal").unwrap_or(false);

        let dropout_p = float_arg("dropout_p").unwrap_or(0.0) as f32;
        anyhow::ensure!(
            dropout_p == 0.0,
            "SDPA: dropout_p={dropout_p} unsupported (inference only)"
        );

        // Default scale 1/sqrt(head_dim) needs a concrete head_dim;
        // symbolic-shape HF graphs always pass `scale` explicitly.
        let scale = match float_arg("scale") {
            Some(v) => v,
            None => {
                let head_dim = query.shape.dims.last().and_then(|d| d.to_usize()).context(
                    "SDPA: query head_dim must be concrete to derive the default \
                         scale (pass `scale` explicitly for symbolic head dims)",
                )?;
                1.0_f64 / (head_dim as f64).sqrt()
            }
        };

        // GQA: repeat-interleave K/V heads up to Q's count, gated on
        // `enable_gqa` — only the unified op carries the flag, and the
        // pipeline emits only the unified op (sdpa decomps are stripped).
        // Flagless graphs with mismatched heads fail the ensure below.
        let enable_gqa = bool_arg("enable_gqa").unwrap_or(false);
        let mut gqa_group: Option<Expression> = None;
        if enable_gqa && q_ndim >= 3 && query.shape.dims[q_ndim - 3] != key.shape.dims[q_ndim - 3] {
            let h_axis = q_ndim - 3;
            let h_q = query.shape.dims[h_axis]
                .to_usize()
                .context("SDPA GQA: query head count must be concrete")?;
            let h_kv = key.shape.dims[h_axis]
                .to_usize()
                .context("SDPA GQA: kv head count must be concrete")?;
            anyhow::ensure!(
                h_kv > 0 && h_q % h_kv == 0,
                "SDPA GQA: query heads ({h_q}) must be a positive multiple of kv heads ({h_kv})"
            );
            gqa_group = Some(Expression::from(h_q / h_kv));
        } else if q_ndim >= 3 {
            anyhow::ensure!(
                query.shape.dims[q_ndim - 3] == key.shape.dims[q_ndim - 3],
                "SDPA: query/key head counts differ ({:?} vs {:?}) without enable_gqa",
                query.shape.dims[q_ndim - 3],
                key.shape.dims[q_ndim - 3]
            );
        }

        // ── The native attention spelling ──
        //
        // From here on the chain is spelled the way luminal's hand-written
        // paged models spell it (`examples/paged_llama`, and the serving
        // engine's llama3), so that the SAME e-graph rules — the FlashInfer
        // islands in `luminal_cuda_lite` — see the same nodes whether the
        // graph was written by hand or translated:
        //
        // * leading unit dims are squeezed away (a view) so the chain is
        //   rank 3 — `(heads, s, d)`, `(heads, d, c)`, `(heads, s, c)`; the
        //   islands' patterns are rank 3, and a size-1 batch axis is a fourth
        //   dim with a nonzero stride that matches none of them;
        // * K is permuted to `(heads, d, c)` and V kept `(heads, c, d)`
        //   BEFORE their GQA expand, and each gets a `* 1.0` contiguity
        //   barrier after `expand_dim + merge_dims` — the barrier is on the
        //   operand the matmul consumes, with the group replication in its
        //   merged-axis stride;
        // * q is re-materialised TOKEN-major (`(s, heads, d)` in memory,
        //   viewed back as `(heads, s, d)`): the FlashInferAttention host op
        //   reads q as `(tokens, heads, dim)`;
        // * Q/K are upcast to F32 after their barriers so QK^T, scale, mask
        //   and softmax are F32 (torch parity, unchanged); V is upcast so P.V
        //   runs in F32 too (previously probs were rounded to the value dtype
        //   — strictly more precise now); the output is cast back to the
        //   value dtype right after P.V. The hand-written models spell the
        //   same casts.
        let (mut q, mut k, mut v) = (query, key, value);
        let mut squeezed = 0usize;
        while q.shape.len() > 3
            && q.shape.dims[0].to_usize() == Some(1)
            && k.shape.dims[0].to_usize() == Some(1)
            && v.shape.dims[0].to_usize() == Some(1)
        {
            q = q.squeeze(0);
            k = k.squeeze(0);
            v = v.squeeze(0);
            squeezed += 1;
        }
        let n = q.shape.len();
        let mut perm: Vec<usize> = (0..n).collect();
        perm.swap(n - 2, n - 1);
        // K: (…, kvh, c, d) -> (…, kvh, d, c), then GQA, then the barrier.
        let mut k_t = k.permute(perm.clone());
        if let Some(group) = gqa_group {
            let h_axis = n - 3;
            k_t = k_t
                .expand_dim(h_axis + 1, group)
                .merge_dims(h_axis, h_axis + 1);
            v = v
                .expand_dim(h_axis + 1, group)
                .merge_dims(h_axis, h_axis + 1);
        }
        let k_t = k_t * 1.0;
        let v = v * 1.0;
        // q: token-major buffer, viewed heads-major.
        let mut q_perm: Vec<usize> = (0..n).collect();
        q_perm.swap(n - 3, n - 2);
        let q = (q.permute(q_perm.clone()) * 1.0).permute(q_perm);
        let (q_for_mm, k_for_mm) = ensure_same_dtype(q, k_t);
        let low_precision = matches!(q_for_mm.dtype, DType::Bf16 | DType::F16);
        // torch parity: fused kernels accumulate QK^T in fp32 and never
        // materialise low-precision scores (CPU: opmath_type = f32, see
        // pytorch's FlashAttentionKernel.cpp; CUDA likewise via tensor-core
        // fp32 accumulators — FA2 paper, arXiv:2307.08691 §3). Q/K are cast
        // AFTER their barriers, so the barriers stay in the input dtype and
        // are what a fused op consumes; scale, mask and softmax are F32.
        let (q_for_mm, k_for_mm) = if low_precision {
            (q_for_mm.cast(DType::F32), k_for_mm.cast(DType::F32))
        } else {
            (q_for_mm, k_for_mm)
        };
        let mut scores = self.apply_scalar_op(q_for_mm.matmul(k_for_mm), scale, BinaryOp::Mul);
        let value = v;
        let q_ndim = n;

        if is_causal {
            let s_q = scores.shape.dims[q_ndim - 2];
            let s_k = scores.shape.dims[q_ndim - 1];
            let row = self.graph.arange(s_q).cast(DType::F32).expand_dim(1, s_k);
            let col = self.graph.arange(s_k).cast(DType::F32).expand_dim(0, s_q);
            // 1.0 strictly above the diagonal (j > i = masked); -1e9 ≈ -inf.
            let masked = col.gt(row).cast(DType::F32);
            scores = add_offset(scores, masked * (-1e9_f32));
        }

        // Bool masks: track per-row any-keep to zero fully-masked rows
        // after softmax (see doc comment).
        let mut row_any_keep: Option<GraphTensor> = None;
        if let Some(mut mask) = additive {
            while mask.shape.len() > scores.shape.len() && mask.shape.dims[0].to_usize() == Some(1)
            {
                mask = mask.squeeze(0);
            }
            let offset = if mask.dtype == DType::Bool {
                let keep = mask.cast(DType::F32);
                let key_axis = keep.shape.len() - 1;
                row_any_keep = Some(
                    keep.max(key_axis)
                        .expand_to_shape_on_axes(keep.shape, key_axis),
                );
                let one = keep.graph().constant_float(1.0).expand_rhs(keep.shape);
                (one - keep) * -1e9_f32
            } else {
                mask
            };
            scores = add_offset(scores, offset);
        }

        // Tuple outputs serialize as one `as_tensors` list or one entry per
        // element — flatten to slot order; slot 0 is the attention output.
        let output_names: Vec<String> = node
            .outputs
            .iter()
            .flat_map(|o| {
                if let Some(t) = o.as_tensor.as_ref() {
                    vec![t.name.clone()]
                } else if let Some(ts) = o.as_tensors.as_ref() {
                    ts.iter().map(|t| t.name.clone()).collect()
                } else {
                    vec![]
                }
            })
            .collect();

        let mut attn = scores.softmax(q_ndim - 1);
        if let Some(indicator) = row_any_keep {
            let (a, i) = ensure_same_dtype(attn, indicator);
            let (a, i) = broadcast_binary(a, i);
            attn = a * i;
        }
        // P.V in F32 (probs stay F32, V upcast), cast back to the value dtype
        // right after — where the hand-written models cast.
        let out_dtype = value.dtype;
        let v_mm = if low_precision {
            value.cast(DType::F32)
        } else {
            value
        };
        let mut out = attn.matmul(v_mm);
        if low_precision {
            out = out.cast(out_dtype);
        }
        for _ in 0..squeezed {
            out = out.unsqueeze(0);
        }

        match output_names.first().filter(|n| !n.is_empty()) {
            Some(name) => {
                self.tensors.insert(name.clone(), out);
                Ok(())
            }
            None => anyhow::bail!("SDPA: no output tensor name found on node {}", node.target),
        }
    }
}
