# The Rock8 — features

**This file is no longer maintained separately. See [README.md](README.md).**

It previously duplicated the feature tables from the README, and the two drifted:
by the time that was noticed, this file still attributed the "95 t/s > Vulkan 91"
decode win to the wrong model, still quoted a superseded batching figure, and
still listed an operations feature (`roc8-autoctx`) that exists in no source file
in any of our repositories. A reader who happened to open this one instead of the
README got three wrong answers.

Rather than re-synchronise two hand-maintained copies and wait for them to drift
again, the README is now the single source. It carries everything this file had,
plus the full weight-format table, the prefill-route environment variables, the
correctness fixes, the published models, and the licensing notes — none of which
were ever here.

- **Features, formats, environment variables:** [README.md](README.md)
- **Getting an RDNA4 card running:** [README.md § 0](README.md#0-get-an-rdna4-card-working--start-here)
- **The kernels themselves:** [The-Monk/llama.cpp, branch `roc8`](https://github.com/The-Monk/llama.cpp/tree/roc8)
