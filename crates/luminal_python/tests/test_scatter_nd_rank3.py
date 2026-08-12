"""Regression test: a 3-D indexed scatter must write the whole block.

  LUMINAL_PYTHON_SRC=<luminal>/crates/luminal_python/src \
      python3 src/translated/probe_miscompile.py

`pool[idx, :, :] = v` on a 3-D tensor used to store only the first element along
the LAST dimension — 32 values where there should be 256 on a [64, 8, 8] pool.
The 2-D form was correct and so was the matching 3-D read, which is why it hid:
a rank-2 target never exercises the broken path. Fixed in
`movement_dynamic::pt2_scatter_nd`.

This is why a paged KV cache produces garbage: transformers stores its pools as
[num_slots, num_kv_heads, head_dim] and writes them with exactly this
expression, so 1/head_dim of each entry survives and the rest of the context is
whatever the pool held before.

No model, no transformers, no inference engine.
"""

import functools
import os
import sys

import torch
from torch import nn

if os.environ.get("LUMINAL_PYTHON_SRC"):
    sys.path.insert(0, os.environ["LUMINAL_PYTHON_SRC"])
from luminal import luminal_backend  # noqa: E402

DEV = "cuda"
SHAPE_3D = (64, 8, 8)


def compiled(module):
    return torch.compile(
        module, backend=functools.partial(luminal_backend, options={"search_iterations": 1})
    )


class Scatter3D(nn.Module):
    def forward(self, pool, index, values):
        pool[index, :, :] = values
        return pool * 1.0


class Scatter2D(nn.Module):
    def forward(self, pool, index, values):
        pool[index, :] = values
        return pool * 1.0


class Gather3D(nn.Module):
    def forward(self, pool, index):
        return pool[index, :, :]


def check(name, module, shape, write):
    def fresh():
        pool = torch.zeros(shape, dtype=torch.float32, device=DEV)
        index = torch.arange(4, dtype=torch.int64, device=DEV)
        if not write:
            pool = torch.arange(
                pool.numel(), dtype=torch.float32, device=DEV
            ).reshape(shape)
            return pool, index
        return pool, index, torch.full((4,) + shape[1:], 7.0, device=DEV)

    args = fresh()
    with torch.no_grad():
        gold = module(*(a.clone() if i == 0 else a for i, a in enumerate(args))).cpu()
    run = compiled(module)
    args = fresh()
    torch._dynamo.mark_dynamic(args[1], 0)
    with torch.no_grad():
        got = run(*args).cpu()
    ok = torch.equal(gold, got)
    print(
        f"  {name:24} correct={str(ok):5}  "
        f"nonzeros eager={int((gold != 0).sum()):5} compiled={int((got != 0).sum()):5}"
    )
    return ok


def main():
    print("\n=== 3-D indexed scatter ===")
    ok = [
        check("scatter 3D pool[i,:,:]", Scatter3D(), SHAPE_3D, True),
        check("scatter 2D pool[i,:]", Scatter2D(), (64, 8), True),
        check("gather  3D pool[i,:,:]", Gather3D(), SHAPE_3D, False),
    ]
    print("\nEXPECTED: all three pass.")
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
