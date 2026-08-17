# vendor/

`egglog-add-subsumed.patch` — the local engine addition the vendored
substitution walk needs: `Write::add_subsumed` /
`TableAction::lookup_or_insert_subsumed`, an atomic insert-born-retired
(a staged insert cannot be re-subsumed through the public API, and even
a one-round live window for a copied retired spelling re-arms the
orbits the `:subsume` termination discipline prevents).

The patch is applied to a checkout of egglog @ c2c0f151 living at
`vendor/egglog-checkout (gitignored)`, referenced by the
`[patch]` section in the workspace `Cargo.toml`. BEFORE this branch
goes anywhere shared, the fork/upstream ask still stands; meanwhile every machine clones+patches into the SAME relative path — a luminal-ai
egglog fork or the upstream ask (proposed alongside
egglog-experimental #60). Regenerate the checkout by cloning egglog at
c2c0f151 and applying this patch file.
