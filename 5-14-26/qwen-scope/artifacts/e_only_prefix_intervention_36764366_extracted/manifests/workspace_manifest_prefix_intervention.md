# 5-14-26 Qwen-Scope Prefix Intervention Workspace

Fresh remote workspace for prefix-level behavioral intervention on the e_only stream path.

This is not residual steering and not SAE feature steering. It uses Transformers/PyTorch selected-layer capture plus Qwen-Scope TopK-50 SAE encoding.

- instance_id: `36764366`
- model_repo: `Qwen/Qwen3.5-35B-A3B-Base`
- model_path: `/workspace/qwen-scope/5-14-26/models/Qwen3.5-35B-A3B-Base`
- sae_repo: `Qwen/SAE-Res-Qwen3.5-35B-A3B-Base-W32K-L0_50`
- sae_path: `/workspace/qwen-scope/5-14-26/saes/SAE-Res-Qwen3.5-35B-A3B-Base-W32K-L0_50`
- restrictions: no residual steering, no SAE feature steering, no Hauhau, no llama.cpp, no all-layer expansion, no semantic labels
