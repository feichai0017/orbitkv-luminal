//! String-backed symbolic-dimension names (our landing of PR #396's
//! design, ruling 2026-08-13): `Symbol` is a Copy handle into a
//! process-global interner, so `Term` stays Copy while names are
//! arbitrary-length. Names validate against `[A-Za-z][A-Za-z0-9_]*`
//! with no doubled underscore and are REJECTED, never sanitized
//! (sanitizing is not injective — "a.b" and "a-b" must not collide).
//! The alphabet guarantees by construction that a name is a valid
//! egglog string literal and C identifier, so no codegen site
//! re-checks. Unlike main's PR, NO name is reserved: this branch
//! retired 'z' (z-var retirement, 2026-08-06) — every name is an
//! ordinary symbol.
//!
//! Equality/hash are by interner index (interning makes name-equality
//! coincide with index-equality); Ord is BY NAME, so any
//! order-dependent downstream (slot assignment on real backends) is
//! deterministic in the name vocabulary, not in interning order.

use rustc_hash::FxHashMap;
use std::sync::{OnceLock, RwLock};

struct Interner {
    names: Vec<&'static str>,
    index: FxHashMap<&'static str, u32>,
    fresh: u64,
}

fn interner() -> &'static RwLock<Interner> {
    static INTERNER: OnceLock<RwLock<Interner>> = OnceLock::new();
    INTERNER.get_or_init(|| {
        RwLock::new(Interner {
            names: Vec::new(),
            index: FxHashMap::default(),
            fresh: 0,
        })
    })
}

fn is_well_formed(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    first.is_ascii_alphabetic()
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !name.contains("__")
}

/// An interned symbolic-dimension name.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Symbol(u32);

impl Symbol {
    /// Intern a validated name — panics loudly on malformed input.
    pub fn new(name: impl AsRef<str>) -> Self {
        let name = name.as_ref();
        assert!(
            is_well_formed(name),
            "symbol name {name:?} must match [A-Za-z][A-Za-z0-9_]* with no \
             doubled underscore (reject, never sanitize)"
        );
        Self::intern(name)
    }

    fn intern(name: &str) -> Self {
        if let Some(&idx) = interner().read().unwrap().index.get(name) {
            return Symbol(idx);
        }
        let mut guard = interner().write().unwrap();
        if let Some(&idx) = guard.index.get(name) {
            return Symbol(idx);
        }
        let leaked: &'static str = Box::leak(name.to_string().into_boxed_str());
        let idx = guard.names.len() as u32;
        guard.names.push(leaked);
        guard.index.insert(leaked, idx);
        Symbol(idx)
    }

    /// A fresh symbol no prior name can collide with — replaces the old
    /// private-use-char trick for internal temporaries.
    pub fn fresh(stem: &str) -> Self {
        let n = {
            let mut guard = interner().write().unwrap();
            guard.fresh += 1;
            guard.fresh
        };
        Self::new(format!("{stem}{n}"))
    }

    pub fn name(&self) -> &'static str {
        interner().read().unwrap().names[self.0 as usize]
    }
}

impl PartialOrd for Symbol {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Symbol {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.name().cmp(other.name())
    }
}

impl std::fmt::Display for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

impl std::fmt::Debug for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

impl From<char> for Symbol {
    fn from(c: char) -> Self {
        Symbol::new(c.to_string())
    }
}

impl From<&char> for Symbol {
    fn from(c: &char) -> Self {
        Symbol::from(*c)
    }
}

impl From<&str> for Symbol {
    fn from(s: &str) -> Self {
        Symbol::new(s)
    }
}

impl serde::Serialize for Symbol {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.name())
    }
}

impl<'de> serde::Deserialize<'de> for Symbol {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let name = String::deserialize(deserializer)?;
        if !is_well_formed(&name) {
            return Err(serde::de::Error::custom(format!(
                "unusable symbol name {name:?}"
            )));
        }
        Ok(Symbol::intern(&name))
    }
}

/// The dynamic-dimension binding map (PR #396 vocabulary).
pub type DynMap = FxHashMap<Symbol, usize>;

#[cfg(test)]
mod tests {
    use super::Symbol;

    #[test]
    fn interning_equality_and_name_order() {
        let a = Symbol::new("seq");
        let b = Symbol::from("seq");
        assert_eq!(a, b);
        assert_eq!(a.name(), "seq");
        assert!(Symbol::new("a") < Symbol::new("b"), "Ord is by name");
        assert_eq!(Symbol::from('s').name(), "s");
    }

    #[test]
    #[should_panic(expected = "reject, never sanitize")]
    fn malformed_names_are_rejected() {
        Symbol::new("a.b");
    }

    #[test]
    fn fresh_symbols_never_collide() {
        assert_ne!(Symbol::fresh("tmp"), Symbol::fresh("tmp"));
    }

    #[test]
    fn serde_round_trips_by_name() {
        let s = Symbol::new("seq_len");
        let json = serde_json::to_string(&s).unwrap();
        assert_eq!(json, "\"seq_len\"");
        let back: Symbol = serde_json::from_str(&json).unwrap();
        assert_eq!(back, s);
    }
}
