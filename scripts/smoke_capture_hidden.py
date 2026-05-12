#!/usr/bin/env python3
"""Minimal Qwen-Scope hidden-state smoke capture.

Loads the local Hugging Face base model, runs one tiny prompt, and saves the
final prompt-token residual vectors for a small layer subset only.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path("/workspace/qwen-scope/5-11-26")
MODEL_PATH = ROOT / "models" / "Qwen3.5-35B-A3B-Base"
OUT_DIR = Path(os.environ.get("SMOKE_CAPTURE_HIDDEN_OUT_DIR", ROOT / "hidden_states" / "smoke"))
OFFLOAD_DIR = Path(os.environ.get("SMOKE_CAPTURE_OFFLOAD_DIR", ROOT / ".offload" / "smoke_capture_hidden"))
PROMPT = "What is 2+2? Answer briefly."
SELECTED_LAYERS = [14, 26]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def tensor_summary(tensor: torch.Tensor) -> dict[str, object]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "device": str(tensor.device),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)

    started_at = utc_now()
    print(f"started_at={started_at}")
    print(f"model_path={MODEL_PATH}")
    print(f"output_dir={OUT_DIR}")
    print(f"prompt={PROMPT!r}")
    print(f"selected_layers={SELECTED_LAYERS}")
    print(f"torch={torch.__version__}")
    print(f"torch_cuda={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"cuda_device_count={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(
            "cuda_devices="
            + json.dumps([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    rendered_prompt = PROMPT
    encoded = tokenizer(rendered_prompt, return_tensors="pt")
    token_count = int(encoded["input_ids"].shape[1])
    final_prompt_token_index = token_count - 1
    print(f"token_count={token_count}")
    print(f"final_prompt_token_index={final_prompt_token_index}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        device_map="auto",
        dtype=torch.bfloat16,
        offload_folder=str(OFFLOAD_DIR),
        offload_state_dict=True,
    )
    model.eval()
    print("model_loaded=true")
    print("model_class=" + model.__class__.__name__)
    print("hf_device_map=" + json.dumps(getattr(model, "hf_device_map", {}), sort_keys=True))

    input_device = model.get_input_embeddings().weight.device
    encoded = {key: value.to(input_device) for key, value in encoded.items()}

    with torch.inference_mode():
        generated_ids = model.generate(
            **encoded,
            max_new_tokens=16,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        new_token_ids = generated_ids[0, token_count:]
        generated_text = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
        print("generated_text=" + json.dumps(generated_text))

        outputs = model(
            **encoded,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("Model forward pass did not return hidden_states")

    text_config = getattr(model.config, "text_config", model.config)
    hidden_size = int(getattr(text_config, "hidden_size"))
    num_hidden_layers = int(getattr(text_config, "num_hidden_layers"))
    expected_tuple_len = num_hidden_layers + 1
    print(f"hidden_size={hidden_size}")
    print(f"num_hidden_layers={num_hidden_layers}")
    print(f"hidden_state_tuple_len={len(hidden_states)}")
    print(f"expected_hidden_state_tuple_len={expected_tuple_len}")

    if len(hidden_states) != expected_tuple_len:
        raise RuntimeError(
            f"Expected {expected_tuple_len} hidden-state entries, got {len(hidden_states)}"
        )

    saved_layers = []
    for layer_idx in SELECTED_LAYERS:
        if layer_idx < 0 or layer_idx >= num_hidden_layers:
            raise ValueError(f"Layer index {layer_idx} outside 0..{num_hidden_layers - 1}")

        tuple_index = layer_idx + 1
        layer_hidden = hidden_states[tuple_index]
        vector = layer_hidden[0, final_prompt_token_index, :].detach().to("cpu", dtype=torch.float32)
        if vector.shape != (hidden_size,):
            raise RuntimeError(
                f"Layer {layer_idx} vector shape {tuple(vector.shape)} != ({hidden_size},)"
            )

        out_path = OUT_DIR / f"layer_{layer_idx:02d}_final_prompt_resid.pt"
        payload = {
            "layer_index": layer_idx,
            "hf_hidden_states_tuple_index": tuple_index,
            "layer_indexing_note": (
                "Saved decoder layer output for zero-based layer_index; "
                "Transformers hidden_states[0] is embeddings, so tuple index is layer_index + 1."
            ),
            "prompt_token_index": final_prompt_token_index,
            "prompt_token_id": int(encoded["input_ids"][0, final_prompt_token_index].item()),
            "vector": vector,
            "vector_shape": list(vector.shape),
            "vector_dtype": str(vector.dtype).replace("torch.", ""),
        }
        torch.save(payload, out_path)
        saved_layers.append(
            {
                "layer_index": layer_idx,
                "hf_hidden_states_tuple_index": tuple_index,
                "path": str(out_path),
                "vector_shape": list(vector.shape),
                "vector_dtype": str(vector.dtype).replace("torch.", ""),
            }
        )
        print(f"saved_layer={layer_idx} path={out_path} shape={list(vector.shape)}")

    metadata = {
        "started_at": started_at,
        "completed_at": utc_now(),
        "purpose": "minimal Qwen-Scope residual hidden-state smoke capture",
        "phase": "Transformers/PyTorch residual-stream capture; not llama.cpp/GGUF/router capture",
        "model_path": str(MODEL_PATH),
        "output_dir": str(OUT_DIR),
        "rendered_prompt": rendered_prompt,
        "token_count": token_count,
        "final_prompt_token_index": final_prompt_token_index,
        "selected_layers": SELECTED_LAYERS,
        "layer_indexing_note": (
            "Selected layers are zero-based decoder layer indices. Saved tensors use "
            "Transformers hidden_states[layer_index + 1] because hidden_states[0] is embeddings."
        ),
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "generated_text": generated_text,
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_devices": [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
            if torch.cuda.is_available()
            else [],
        },
        "tokenizer": {
            "class": tokenizer.__class__.__name__,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        },
        "model": {
            "class": model.__class__.__name__,
            "device_map": getattr(model, "hf_device_map", {}),
            "input_embedding_device": str(input_device),
        },
        "input_ids_summary": tensor_summary(encoded["input_ids"]),
        "saved_layers": saved_layers,
    }

    metadata_path = OUT_DIR / "smoke_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"metadata_path={metadata_path}")
    print("smoke_capture_hidden_status=ok")


if __name__ == "__main__":
    main()
