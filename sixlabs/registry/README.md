# The model registry

One file, `models.json`, says for every model 6ix.cpp knows about: which family serves it, which
checkpoints it accepts, how it is launched, and how far it has been verified. `registry.py` checks
that file against the tree, prints the status and speed table, and turns an entry into the
deployment JSON `scripts/dgpp-cluster` takes.

```
python3 sixlabs/registry/registry.py check                 # registry vs the tree; exit 1 on any problem
python3 sixlabs/registry/registry.py table                 # every model, 6ix.cpp beside its baselines
python3 sixlabs/registry/registry.py table --status working,compiled
python3 sixlabs/registry/registry.py show qwen3.6-35b-a3b
python3 sixlabs/registry/registry.py resolve qwen3.6-35b-a3b            # writes deploy/cluster_<model>_<format>_w<N>.json
python3 sixlabs/registry/registry.py launch qwen3.6-35b-a3b --dry-run   # resolve, then `dgpp-cluster up --config …`
python3 -m unittest sixlabs/registry/test_registry.py
```

## Why

- A model's settings lived in three places (the template, someone's memory, the fleet's recipes).
  The 35B was first measured with a prefill default that made it look slower than the engine is.
- Ports land on main before they have run on a GPU. Nothing said so except the commit message.
- Adding a family to the serve binary without saying what it is for should fail a check.

## Status

A checkpoint has exactly one status; a model is as far along as its best checkpoint.

| status | meaning | `resolve` / `launch` |
|---|---|---|
| `working` | verified on a GPU on this fleet: bound, passed the forward-check gate, served, measured. Has a record in the tree | yes |
| `upstream` | served by upstream DGPP with a template and published measurements; not verified here | yes |
| `compiled` | on main and builds on a Spark; never loaded weights | only with `--allow-unverified` |
| `groundwork` | config, tensor table and references only; no kernel, loader or assembly | never |
| `refused` | a checkpoint the engine refuses by name | never |

To move a checkpoint to `working`: run its `gpu_steps`, keep the record in the tree, then fill in
`verified` (date, box, engine commit, record) and a `measured` block. `check` will not accept
`working` without them.

## Format

```
families.<name>
  architectures     prefixes of the checkpoint's `architectures[0]` that select this family
  served            true if the serve binary has this family (check compares with apps/dgpp_serve.cpp)
  on_main           false while the family lives on a branch
  tokenizer         which tokenizer implementation it needs (bpe, spm, …)
  engine_defaults   engine keys to use when a deployment template does not set them

models.<id>
  title, family
  text              tool_calls (json | xml-function | checkpoint-template), thinking, vision (served | bound-not-served | none)
  fleet.alias       the gateway name, if the fleet serves it
  branch            set while the model's code is not on main
  checkpoints[]
    repo, revision  the Hugging Face repository and the revision the tensor table was checked against
    format          one of `formats`
    status          see above; exactly one checkpoint has "default": true
    tensors         the binding table's count for that revision
    gpu_steps       the verification steps (required for compiled)
    verified        date, box, engine_commit, record (required for working)
    notes           required for groundwork and refused: why
    worlds.<N>      N Sparks: template, variants{name: template}, engine{overrides}, measured{…}
  baselines[]       the same model on another engine: engine, source, and the measured keys
```

`measured` and `baselines` share keys: `date`, `box`, `source`, `load_cold_s`, `load_warm_s`,
`plan_gib`, `gpu_gib`, and `tok_s` with `single_prose`, `single_code`, `c2`, `c4`, `c8`. A value
is a number, or `[low, high]` where the source gives a range. Every figure names its source; a
figure that was published by someone else and not measured here says so.

## What resolve does

Precedence, lowest first: the checkpoint's template for that world → the family's
`engine_defaults`, for keys the template does not set → the world's `engine` overrides →
`--set key=value`. The output has exactly the template's top-level keys, so the engine's config
parser sees nothing it does not know. Site copies are written as `deploy/cluster_*.json`, which
git ignores.

## What check compares

- every family marked `served` is a family name in `apps/dgpp_serve.cpp`, and the reverse;
- every architecture prefix of an on-main family is recognised in `src/loaders/architecture.cpp`;
- every `engine_defaults` key is one `src/serve/cluster_config.cpp` parses;
- every template and variant exists, is JSON, names that checkpoint and that world size;
- `working` has a record that exists and a measurement; `compiled` has its steps;
- one default checkpoint per model; unique fleet aliases; nothing on a branch is marked runnable.

## Not done yet

- The serve binary does not read the registry: it still picks the family from the checkpoint's
  config, and per-family defaults are applied by `resolve`, not by the engine.
- No engine build is pinned per model. `verified.engine_commit` records what a model was verified
  on; nothing yet stops a newer build from serving it unverified.
- The fleet's own tables (the 6ixlabs repo's recipes, slots and gateway aliases) are not checked
  against this file.
