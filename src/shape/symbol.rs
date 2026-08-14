//! String-backed symbolic-dimension names (our landing of PR #396's
//! design, ruling 2026-08-13): `Symbol` is a Copy handle — a
//! `GenerationalBox<String, SyncStorage>` owned by a process-global
//! owner, the same generational_box pattern `IntegerExpression` uses
//! for its terms — so `Term` stays Copy while names are
//! arbitrary-length. Names validate against `[A-Za-z][A-Za-z0-9_]*`
//! with no doubled underscore and are REJECTED, never sanitized
//! (sanitizing is not injective — "a.b" and "a-b" must not collide).
//! The alphabet guarantees by construction that a name is a valid
//! egglog string literal and C identifier, so no codegen site
//! re-checks. Unlike main's PR, NO name is reserved: this branch
//! retired 'z' (z-var retirement, 2026-08-06) — every name is an
//! ordinary symbol.
//!
//! Equality/hash read THROUGH the handle and compare/hash the NAME
//! string (never the arena slot + generation — derived impls would
//! break the moment two handles named the same thing); Ord is BY
//! NAME, so any order-dependent downstream (slot assignment on real
//! backends) is deterministic in the name vocabulary, not in
//! interning order. Construction still interns — one arena slot per
//! distinct name — but that is a leak-avoidance optimization, not a
//! correctness requirement.

use generational_box::{AnyStorage, GenerationalBox, GenerationalBoxId, Owner, SyncStorage};
use rustc_hash::FxHashMap;
use std::sync::{
    OnceLock, RwLock,
    atomic::{AtomicU64, Ordering},
};

type NameBox = GenerationalBox<String, SyncStorage>;

static NAME_OWNER: OnceLock<Owner<SyncStorage>> = OnceLock::new();
static NAME_INTERNER: OnceLock<RwLock<FxHashMap<String, NameBox>>> = OnceLock::new();
/// One bounded leak per interned name so `name()` can keep handing out
/// `&'static str` (exactly the leak the old `u32`-index interner made).
static LEAKED_NAMES: OnceLock<RwLock<FxHashMap<GenerationalBoxId, &'static str>>> = OnceLock::new();
static FRESH_COUNTER: AtomicU64 = AtomicU64::new(0);

fn interner() -> &'static RwLock<FxHashMap<String, NameBox>> {
    NAME_INTERNER.get_or_init(|| RwLock::new(FxHashMap::default()))
}

fn leaked_names() -> &'static RwLock<FxHashMap<GenerationalBoxId, &'static str>> {
    LEAKED_NAMES.get_or_init(|| RwLock::new(FxHashMap::default()))
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
#[derive(Clone, Copy)]
pub struct Symbol(NameBox);

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
        // Fast path: the name is already interned (read lock only).
        if let Some(&existing) = interner().read().unwrap().get(name) {
            return Symbol(existing);
        }
        // Slow path: insert (write lock), double-checked because another
        // thread may have interned the name between the two locks.
        let mut guard = interner().write().unwrap();
        if let Some(&existing) = guard.get(name) {
            return Symbol(existing);
        }
        let box_ = NAME_OWNER
            .get_or_init(SyncStorage::owner)
            .insert(name.to_string());
        leaked_names()
            .write()
            .unwrap()
            .insert(box_.id(), Box::leak(name.to_string().into_boxed_str()));
        guard.insert(name.to_string(), box_);
        Symbol(box_)
    }

    /// A fresh symbol no prior name can collide with — replaces the old
    /// private-use-char trick for internal temporaries.
    pub fn fresh(stem: &str) -> Self {
        let n = FRESH_COUNTER.fetch_add(1, Ordering::Relaxed) + 1;
        Self::new(format!("{stem}{n}"))
    }

    pub fn name(&self) -> &'static str {
        leaked_names().read().unwrap()[&self.0.id()]
    }
}

impl PartialEq for Symbol {
    fn eq(&self, other: &Self) -> bool {
        // Same arena slot is definitionally the same name; otherwise
        // compare the name strings through the handles.
        self.0.ptr_eq(&other.0) || *self.0.read() == *other.0.read()
    }
}

impl Eq for Symbol {}

impl std::hash::Hash for Symbol {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.read().hash(state);
    }
}

impl PartialOrd for Symbol {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Symbol {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.0.ptr_eq(&other.0) {
            return std::cmp::Ordering::Equal;
        }
        self.0.read().cmp(&other.0.read())
    }
}

impl std::fmt::Display for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0.read())
    }
}

impl std::fmt::Debug for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0.read())
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
        serializer.serialize_str(&self.0.read())
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
