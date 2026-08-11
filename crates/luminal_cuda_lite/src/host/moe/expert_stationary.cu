// Expert-stationary MoE phase 1: gate_up dequant-GEMM + clamped SwiGLU.
//
// decode.cu parallelizes over (pair, row-block): every (token, expert) pair
// re-reads the expert's weights, so weight traffic is pairs x 12.4 MB. Here a
// block owns (expert, row-tile) and streams the tokens routed to that expert
// through weights held in registers, so each expert's weights are read once
// per row-tile regardless of how many tokens routed to it.
//
// Two changes, independently tunable via the template params:
//   ES_M  tokens per tile -> weight reuse    (ES_M=1 reproduces decode.cu's
//                                             traffic, isolating the layout)
//   ES_R  rows per warp   -> activation reuse
//
// LAYOUT. decode.cu gives each lane a whole 32-column scale group, so the
// activation float4s are strided 128 B across the warp: 32 distinct cache
// lines per load instruction. Here a lane owns two 4-column chunks 128
// columns apart, so both the activation float4s (16 B, lane-stride 16 B) and
// the packed-fp4 ushorts (2 B, lane-stride 2 B) are contiguous across the
// warp. Same bytes, ~8x fewer LSU wavefronts on the activation side.
//
// Weights: fp4 blocks [E, N, K/2] (lo nibble = even k), e8m0 scales
// [E, N, K/32], bf16 biases. Requires K % 32 == 0 and an even row0.

#include <cuda_bf16.h>

#ifndef MOE_FP4_COMMON
#define MOE_FP4_COMMON

__constant__ float FP4_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

#define STAGE_LUT(name)                       \
    __shared__ float name[16];                \
    if (threadIdx.x < 16) {                   \
        name[threadIdx.x] = FP4_LUT[threadIdx.x]; \
    }                                         \
    __syncthreads();

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, o);
    }
    return v;
}

#endif // MOE_FP4_COMMON

// Columns a warp covers per k-step: 32 lanes x 2 chunks x 4 columns.
#define ES_CHUNK 256

// ── index build ───────────────────────────────────────────────────────────
// Counting sort of the (token, k) pairs by expert. One block; E <= 1024.
// Emits expert_off[E+1] (exclusive scan of counts) and pairs sorted by
// expert (order within an expert is unspecified — it never matters, each
// pair writes its own hidden row).
extern "C" __global__ void moe_build_expert_index(
    unsigned long long topk_ids_ptr, unsigned long long expert_off_ptr,
    unsigned long long sorted_pairs_ptr,
    int seq, int top_k, int idx_row_stride, int num_experts
) {
    const int* topk_ids = (const int*)topk_ids_ptr;
    int* expert_off = (int*)expert_off_ptr;
    int* sorted_pairs = (int*)sorted_pairs_ptr;

    extern __shared__ int smem[];
    int* counts = smem;                 // [E]
    int* cursor = smem + num_experts;   // [E]

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int num_pairs = seq * top_k;

    for (int i = tid; i < num_experts; i += nthreads) counts[i] = 0;
    __syncthreads();

    for (int p = tid; p < num_pairs; p += nthreads) {
        const int t = p / top_k;
        const int e = topk_ids[(long long)t * idx_row_stride + (p % top_k)];
        if (e >= 0 && e < num_experts) atomicAdd(&counts[e], 1);
    }
    __syncthreads();

    // Exclusive scan, serial in one thread: E is 128 and this runs once per
    // MoE call, against ~10 ms of GEMM.
    if (tid == 0) {
        int run = 0;
        for (int e = 0; e < num_experts; ++e) {
            expert_off[e] = run;
            cursor[e] = run;
            run += counts[e];
        }
        expert_off[num_experts] = run;
    }
    __syncthreads();

    for (int p = tid; p < num_pairs; p += nthreads) {
        const int t = p / top_k;
        const int e = topk_ids[(long long)t * idx_row_stride + (p % top_k)];
        if (e >= 0 && e < num_experts) sorted_pairs[atomicAdd(&cursor[e], 1)] = p;
    }
}

// ── phase 1 ───────────────────────────────────────────────────────────────
// grid = (row_tiles, num_experts); block = ES_WARPS * 32.
// Warp w of the block owns rows [tile*ES_WARPS*ES_R + w*ES_R, +ES_R).
template <int ES_M, int ES_R, int ES_WARPS>
__device__ __forceinline__ void phase1_es_body(
    unsigned long long x_ptr, unsigned long long gu_q_ptr,
    unsigned long long gu_scale_ptr, unsigned long long gu_bias_ptr,
    unsigned long long expert_off_ptr, unsigned long long sorted_pairs_ptr,
    unsigned long long hidden_ptr,
    int hidden_dim, int inter, int top_k, float alpha, float limit
) {
    const float* x = (const float*)x_ptr;
    const unsigned char* gu_q = (const unsigned char*)gu_q_ptr;
    const unsigned char* gu_scale = (const unsigned char*)gu_scale_ptr;
    const __nv_bfloat16* gu_bias = (const __nv_bfloat16*)gu_bias_ptr;
    const int* expert_off = (const int*)expert_off_ptr;
    const int* sorted_pairs = (const int*)sorted_pairs_ptr;
    float* hidden = (float*)hidden_ptr;

    STAGE_LUT(slut)

    const int e = blockIdx.y;
    const int start = expert_off[e];
    const int end = expert_off[e + 1];
    if (start >= end) return;

    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int gate_up_n = 2 * inter;
    const int row0 = (blockIdx.x * ES_WARPS + warp) * ES_R;
    if (row0 >= gate_up_n) return;

    // Byte/element offsets this lane reads at every k-step.
    const int c0 = 4 * lane;          // chunk 0 columns [c0, c0+4)
    const int c1 = 128 + 4 * lane;    // chunk 1
    const int q0 = 2 * lane;          // packed-fp4 byte offset of chunk 0
    const int q1 = 64 + 2 * lane;
    const int g0 = lane / 8;          // scale group of chunk 0
    const int g1 = 4 + lane / 8;

    const long long k_half = hidden_dim / 2;
    const long long k_grp = hidden_dim / 32;
    const unsigned char* qbase = gu_q + ((long long)e * gate_up_n + row0) * k_half;
    const unsigned char* sbase = gu_scale + ((long long)e * gate_up_n + row0) * k_grp;

    for (int mbase = start; mbase < end; mbase += ES_M) {
        const int m_here = min(ES_M, end - mbase);

        // Token row each of the ES_M slots reads from.
        const float* xt[ES_M];
#pragma unroll
        for (int m = 0; m < ES_M; ++m) {
            const int pair = sorted_pairs[mbase + (m < m_here ? m : 0)];
            xt[m] = x + (long long)(pair / top_k) * hidden_dim;
        }

        float acc[ES_M][ES_R];
#pragma unroll
        for (int m = 0; m < ES_M; ++m)
#pragma unroll
            for (int r = 0; r < ES_R; ++r) acc[m][r] = 0.0f;

        for (int base = 0; base < hidden_dim; base += ES_CHUNK) {
            const bool v0 = base + c0 < hidden_dim;
            const bool v1 = base + c1 < hidden_dim;

            // Activations held across all ES_R rows (coalesced: lane-stride
            // 16 B within each chunk), weights transient. The mirror order —
            // dequant all ES_R rows, then stream tokens — was measured and is
            // 1.4x slower at every (M, R): it lengthens the dependency chain
            // into the FMAs without buying back registers.
            float4 a[ES_M][2];
#pragma unroll
            for (int m = 0; m < ES_M; ++m) {
                a[m][0] = v0 ? *reinterpret_cast<const float4*>(xt[m] + base + c0)
                             : make_float4(0.f, 0.f, 0.f, 0.f);
                a[m][1] = v1 ? *reinterpret_cast<const float4*>(xt[m] + base + c1)
                             : make_float4(0.f, 0.f, 0.f, 0.f);
            }

#pragma unroll
            for (int r = 0; r < ES_R; ++r) {
                const unsigned char* qrow = qbase + r * k_half + base / 2;
                const unsigned char* srow = sbase + r * k_grp + base / 32;
                const unsigned int w0 =
                    v0 ? *reinterpret_cast<const unsigned short*>(qrow + q0) : 0u;
                const unsigned int w1 =
                    v1 ? *reinterpret_cast<const unsigned short*>(qrow + q1) : 0u;
                // e8m0 == IEEE-754 exponent field: 2^(sc-127) = bits(sc<<23).
                const float s0 = v0 ? __uint_as_float((unsigned int)srow[g0] << 23) : 0.f;
                const float s1 = v1 ? __uint_as_float((unsigned int)srow[g1] << 23) : 0.f;

                const float b00 = slut[w0 & 0xF] * s0, b01 = slut[(w0 >> 4) & 0xF] * s0;
                const float b02 = slut[(w0 >> 8) & 0xF] * s0, b03 = slut[(w0 >> 12) & 0xF] * s0;
                const float b10 = slut[w1 & 0xF] * s1, b11 = slut[(w1 >> 4) & 0xF] * s1;
                const float b12 = slut[(w1 >> 8) & 0xF] * s1, b13 = slut[(w1 >> 12) & 0xF] * s1;

#pragma unroll
                for (int m = 0; m < ES_M; ++m) {
                    float v = acc[m][r];
                    v = fmaf(b00, a[m][0].x, v);
                    v = fmaf(b01, a[m][0].y, v);
                    v = fmaf(b02, a[m][0].z, v);
                    v = fmaf(b03, a[m][0].w, v);
                    v = fmaf(b10, a[m][1].x, v);
                    v = fmaf(b11, a[m][1].y, v);
                    v = fmaf(b12, a[m][1].z, v);
                    v = fmaf(b13, a[m][1].w, v);
                    acc[m][r] = v;
                }
            }
        }

#pragma unroll
        for (int m = 0; m < ES_M; ++m)
#pragma unroll
            for (int r = 0; r < ES_R; ++r) acc[m][r] = warp_reduce_sum(acc[m][r]);

        // ── bias + clamp + SwiGLU epilogue (rows are gate/up interleaved) ──
        if (lane == 0) {
#pragma unroll
            for (int m = 0; m < ES_M; ++m) {
                if (m >= m_here) break;
                const int pair = sorted_pairs[mbase + m];
#pragma unroll
                for (int jj = 0; jj < ES_R / 2; ++jj) {
                    const int rg = 2 * jj, ru = 2 * jj + 1;
                    if (row0 + ru >= gate_up_n) break;
                    float gate = acc[m][rg] +
                        __bfloat162float(gu_bias[(long long)e * gate_up_n + row0 + rg]);
                    float up = acc[m][ru] +
                        __bfloat162float(gu_bias[(long long)e * gate_up_n + row0 + ru]);
                    gate = fminf(gate, limit);
                    up = fminf(fmaxf(up, -limit), limit);
                    const float sig = 1.0f / (1.0f + expf(-alpha * gate));
                    hidden[(long long)pair * inter + row0 / 2 + jj] =
                        (up + 1.0f) * gate * sig;
                }
            }
        }
    }
}

#define ES_PHASE1(NAME, M, R, W)                                               \
    extern "C" __global__ __launch_bounds__(W * 32) void NAME(                 \
        unsigned long long x_ptr, unsigned long long gu_q_ptr,                 \
        unsigned long long gu_scale_ptr, unsigned long long gu_bias_ptr,       \
        unsigned long long expert_off_ptr, unsigned long long sorted_pairs_ptr,\
        unsigned long long hidden_ptr,                                         \
        int hidden_dim, int inter, int top_k, float alpha, float limit         \
    ) {                                                                        \
        phase1_es_body<M, R, W>(x_ptr, gu_q_ptr, gu_scale_ptr, gu_bias_ptr,    \
                                expert_off_ptr, sorted_pairs_ptr, hidden_ptr,  \
                                hidden_dim, inter, top_k, alpha, limit);       \
    }

// ── phase 2 ───────────────────────────────────────────────────────────────
// Same expert-stationary shape over the down projection, but the top_k
// experts of a token all write the same output row. Rather than atomics
// (nondeterministic across runs, which would make greedy decode
// irreproducible), each pair writes its own scaled row of `partial`
// [pairs, hidden] and moe_phase2_es_reduce sums the top_k of them in a fixed
// order. The extra traffic is ~26 MB per layer against 1.06 GB of weights.
template <int ES_M, int ES_R, int ES_WARPS>
__device__ __forceinline__ void phase2_es_body(
    unsigned long long hidden_ptr, unsigned long long dn_q_ptr,
    unsigned long long dn_scale_ptr, unsigned long long dn_bias_ptr,
    unsigned long long expert_off_ptr, unsigned long long sorted_pairs_ptr,
    unsigned long long topk_w_ptr, unsigned long long partial_ptr,
    int hidden_dim, int inter, int top_k
) {
    const float* hid = (const float*)hidden_ptr;
    const unsigned char* dn_q = (const unsigned char*)dn_q_ptr;
    const unsigned char* dn_scale = (const unsigned char*)dn_scale_ptr;
    const __nv_bfloat16* dn_bias = (const __nv_bfloat16*)dn_bias_ptr;
    const int* expert_off = (const int*)expert_off_ptr;
    const int* sorted_pairs = (const int*)sorted_pairs_ptr;
    const float* topk_w = (const float*)topk_w_ptr;
    float* partial = (float*)partial_ptr;

    STAGE_LUT(slut)

    const int e = blockIdx.y;
    const int start = expert_off[e];
    const int end = expert_off[e + 1];
    if (start >= end) return;

    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int row0 = (blockIdx.x * ES_WARPS + warp) * ES_R;
    if (row0 >= hidden_dim) return;

    const int c0 = 4 * lane, c1 = 128 + 4 * lane;
    const int q0 = 2 * lane, q1 = 64 + 2 * lane;
    const int g0 = lane / 8, g1 = 4 + lane / 8;

    const long long k_half = inter / 2;
    const long long k_grp = inter / 32;
    const unsigned char* qbase = dn_q + ((long long)e * hidden_dim + row0) * k_half;
    const unsigned char* sbase = dn_scale + ((long long)e * hidden_dim + row0) * k_grp;

    for (int mbase = start; mbase < end; mbase += ES_M) {
        const int m_here = min(ES_M, end - mbase);

        // Phase 2's "tokens" are pairs: each has its own hidden row.
        const float* hv[ES_M];
#pragma unroll
        for (int m = 0; m < ES_M; ++m) {
            const int pair = sorted_pairs[mbase + (m < m_here ? m : 0)];
            hv[m] = hid + (long long)pair * inter;
        }

        float acc[ES_M][ES_R];
#pragma unroll
        for (int m = 0; m < ES_M; ++m)
#pragma unroll
            for (int r = 0; r < ES_R; ++r) acc[m][r] = 0.0f;

        for (int base = 0; base < inter; base += ES_CHUNK) {
            const bool v0 = base + c0 < inter;
            const bool v1 = base + c1 < inter;

            float4 a[ES_M][2];
#pragma unroll
            for (int m = 0; m < ES_M; ++m) {
                a[m][0] = v0 ? *reinterpret_cast<const float4*>(hv[m] + base + c0)
                             : make_float4(0.f, 0.f, 0.f, 0.f);
                a[m][1] = v1 ? *reinterpret_cast<const float4*>(hv[m] + base + c1)
                             : make_float4(0.f, 0.f, 0.f, 0.f);
            }

#pragma unroll
            for (int r = 0; r < ES_R; ++r) {
                const unsigned char* qrow = qbase + r * k_half + base / 2;
                const unsigned char* srow = sbase + r * k_grp + base / 32;
                const unsigned int w0 =
                    v0 ? *reinterpret_cast<const unsigned short*>(qrow + q0) : 0u;
                const unsigned int w1 =
                    v1 ? *reinterpret_cast<const unsigned short*>(qrow + q1) : 0u;
                const float s0 = v0 ? __uint_as_float((unsigned int)srow[g0] << 23) : 0.f;
                const float s1 = v1 ? __uint_as_float((unsigned int)srow[g1] << 23) : 0.f;

                const float b00 = slut[w0 & 0xF] * s0, b01 = slut[(w0 >> 4) & 0xF] * s0;
                const float b02 = slut[(w0 >> 8) & 0xF] * s0, b03 = slut[(w0 >> 12) & 0xF] * s0;
                const float b10 = slut[w1 & 0xF] * s1, b11 = slut[(w1 >> 4) & 0xF] * s1;
                const float b12 = slut[(w1 >> 8) & 0xF] * s1, b13 = slut[(w1 >> 12) & 0xF] * s1;

#pragma unroll
                for (int m = 0; m < ES_M; ++m) {
                    float v = acc[m][r];
                    v = fmaf(b00, a[m][0].x, v);
                    v = fmaf(b01, a[m][0].y, v);
                    v = fmaf(b02, a[m][0].z, v);
                    v = fmaf(b03, a[m][0].w, v);
                    v = fmaf(b10, a[m][1].x, v);
                    v = fmaf(b11, a[m][1].y, v);
                    v = fmaf(b12, a[m][1].z, v);
                    v = fmaf(b13, a[m][1].w, v);
                    acc[m][r] = v;
                }
            }
        }

#pragma unroll
        for (int m = 0; m < ES_M; ++m)
#pragma unroll
            for (int r = 0; r < ES_R; ++r) acc[m][r] = warp_reduce_sum(acc[m][r]);

        if (lane == 0) {
#pragma unroll
            for (int m = 0; m < ES_M; ++m) {
                if (m >= m_here) break;
                const int pair = sorted_pairs[mbase + m];
                const float w = topk_w[pair];  // [seq, top_k] contiguous
#pragma unroll
                for (int r = 0; r < ES_R; ++r) {
                    if (row0 + r >= hidden_dim) break;
                    const float b =
                        __bfloat162float(dn_bias[(long long)e * hidden_dim + row0 + r]);
                    partial[(long long)pair * hidden_dim + row0 + r] = w * (acc[m][r] + b);
                }
            }
        }
    }
}

#define ES_PHASE2(NAME, M, R, W)                                               \
    extern "C" __global__ __launch_bounds__(W * 32) void NAME(                 \
        unsigned long long hidden_ptr, unsigned long long dn_q_ptr,            \
        unsigned long long dn_scale_ptr, unsigned long long dn_bias_ptr,       \
        unsigned long long expert_off_ptr, unsigned long long sorted_pairs_ptr,\
        unsigned long long topk_w_ptr, unsigned long long partial_ptr,         \
        int hidden_dim, int inter, int top_k                                   \
    ) {                                                                        \
        phase2_es_body<M, R, W>(hidden_ptr, dn_q_ptr, dn_scale_ptr,            \
                                dn_bias_ptr, expert_off_ptr, sorted_pairs_ptr, \
                                topk_w_ptr, partial_ptr, hidden_dim, inter,    \
                                top_k);                                        \
    }

ES_PHASE2(moe_phase2_es_m2_r8, 2, 8, 8)
ES_PHASE2(moe_phase2_es_m4_r8, 4, 8, 8)
ES_PHASE2(moe_phase2_es_m8_r8, 8, 8, 8)
ES_PHASE2(moe_phase2_es_m4_r4, 4, 4, 8)

// out[t][r] = sum over the token's top_k pairs, in pair order.
extern "C" __global__ void moe_phase2_es_reduce(
    unsigned long long partial_ptr, unsigned long long out_ptr,
    int hidden_dim, int top_k, int seq
) {
    const float* partial = (const float*)partial_ptr;
    float* out = (float*)out_ptr;
    const long long total = (long long)seq * hidden_dim;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (long long)gridDim.x * blockDim.x) {
        const long long t = i / hidden_dim;
        const long long r = i % hidden_dim;
        float s = 0.0f;
        for (int k = 0; k < top_k; ++k)
            s += partial[(t * top_k + k) * hidden_dim + r];
        out[i] = s;
    }
}

// ES_M=1 is the layout-only control: same traffic as decode.cu, new access
// pattern. The rest trade registers for weight reuse.
ES_PHASE1(moe_phase1_es_m1_r8, 1, 8, 8)
ES_PHASE1(moe_phase1_es_m2_r8, 2, 8, 8)
ES_PHASE1(moe_phase1_es_m4_r8, 4, 8, 8)
ES_PHASE1(moe_phase1_es_m8_r8, 8, 8, 8)
ES_PHASE1(moe_phase1_es_m4_r4, 4, 4, 8)
ES_PHASE1(moe_phase1_es_m8_r4, 8, 4, 8)
ES_PHASE1(moe_phase1_es_m16_r4, 16, 4, 8)
ES_PHASE1(moe_phase1_es_m8_r2, 8, 2, 8)
