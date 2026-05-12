# Active 2x96GB Teardown Archive Receipt

Remote workspace:

`/workspace/qwen-scope/5-11-26`

Active instance:

`36563002`

Local archive:

`qwen_scope_5-11-26_active_2x96gb_teardown_small_artifacts_20260512.tar.gz`

Archive SHA256:

`3feea5cf1ab667c5cf0403b6f22c2323f520cab474c4ed44e3c4e7964b354ef7`

Archive size:

`2296149` bytes

Included remote artifact classes:

- `prompts/`
- `scripts/`
- `provenance/`
- `manifests/`
- `logs/`
- `sae_outputs/`
- `hidden_states/`
- `smoke-runs/`

Excluded intentionally:

- model weights under `models/`
- SAE checkpoint weights under `saes/`
- virtualenv/cache/offload directories
- remote `.env.hf`

Rationale:

The excluded model and SAE artifacts are large and reconstructible from the recorded Hugging Face repo IDs and manifests. The included artifacts are the small, non-reconstructible run products needed to audit the Qwen-Scope work.
