# Qwen-Scope 5-12-26 Workspace Manifest

This is the fresh 5-12 Qwen-Scope workspace for Transformers/PyTorch residual-stream capture and Qwen-Scope SAE analysis. It does not depend on the destroyed 5-11 instance.

## Workspace Root

`/workspace/qwen-scope/5-12-26`

## Model Dependency

- Repository: `Qwen/Qwen3.5-35B-A3B-Base`
- Local path: `/workspace/qwen-scope/5-12-26/models/Qwen3.5-35B-A3B-Base`
- Format: Hugging Face Transformers/PyTorch safetensors snapshot

## SAE Dependency

- Repository: `Qwen/SAE-Res-Qwen3.5-35B-A3B-Base-W32K-L0_50`
- Local path: `/workspace/qwen-scope/5-12-26/saes/SAE-Res-Qwen3.5-35B-A3B-Base-W32K-L0_50`
- Format: Qwen-Scope SAE `layer*.sae.pt` files

## Scope

This setup is for residual-stream capture and SAE analysis only. No Hauhau, llama.cpp, GGUF, steering, or experiment run was performed during setup.
