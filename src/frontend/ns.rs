//! The namespace path cursor (Stage 2 of the naming plan, ruling
//! 2026-08-12): modules thread an `Ns` down the module tree —
//! candle-VarBuilder-style — and mint leaf tensor labels under it, so
//! labels ARE checkpoint keys by construction and uniqueness falls out
//! of tree addressing (the PyTorch invariants: segments are non-empty
//! and dot-free, so every label parses back to a unique tree address).
//!
//! The low-level path stays free: an importer that already holds
//! state_dict FQNs calls `Graph::named_tensor` with the raw string and
//! never touches `Ns`. Malformed segments are REJECTED, never
//! sanitized (the PR #396 principle: sanitizing is not injective).

/// A dot-joined path in the module tree. `Default`/`root()` is the
/// empty root; `child`/`index` extend it; `leaf` mints a full label.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Ns {
    path: String,
}

impl Ns {
    pub fn root() -> Self {
        Self::default()
    }

    fn checked(segment: &str) -> &str {
        assert!(
            !segment.is_empty() && !segment.contains('.'),
            "namespace segment {segment:?} must be non-empty and dot-free \
             (dots are the path separator; reject, never sanitize)"
        );
        segment
    }

    /// Descend into a named submodule: `ns.child("self_attn")`.
    pub fn child(&self, segment: impl AsRef<str>) -> Self {
        let segment = Self::checked(segment.as_ref());
        Self {
            path: if self.path.is_empty() {
                segment.to_string()
            } else {
                format!("{}.{segment}", self.path)
            },
        }
    }

    /// Descend into a collection slot: `ns.index(3)` — the stringified
    /// integer child (the ModuleList convention, so paths line up with
    /// PyTorch state_dict keys).
    pub fn index(&self, i: usize) -> Self {
        self.child(i.to_string())
    }

    /// Mint the full label for a leaf tensor: `ns.leaf("weight")`.
    pub fn leaf(&self, name: impl AsRef<str>) -> String {
        self.child(name).path
    }

    /// The path itself (for diagnostics and module-level bookkeeping).
    pub fn path(&self) -> &str {
        &self.path
    }
}

#[cfg(test)]
mod tests {
    use super::Ns;

    #[test]
    fn paths_join_like_state_dict_keys() {
        let ns = Ns::root().child("model").child("layers").index(0);
        assert_eq!(
            ns.child("self_attn").leaf("weight"),
            "model.layers.0.self_attn.weight"
        );
        assert_eq!(Ns::root().leaf("lm_head"), "lm_head");
    }

    #[test]
    #[should_panic(expected = "dot-free")]
    fn dotted_segments_are_rejected_never_sanitized() {
        Ns::root().child("model.layers");
    }

    #[test]
    #[should_panic(expected = "non-empty")]
    fn empty_segments_are_rejected() {
        Ns::root().child("");
    }
}
