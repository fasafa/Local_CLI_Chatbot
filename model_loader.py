from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


def load_model_and_tokenizer(model_name: str = "Qwen/Qwen2-0.5B", use_gpu_if_available: bool = True):
    """
    Loads tokenizer and model, returns a Hugging Face text-generation pipeline.
    - model_name: HF model id (default: "google/flan-t5-small")
    - use_gpu_if_available: attempt to use GPU when available
    """
    print(f"[model_loader] Loading tokenizer and model '{model_name}' ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Decide device for pipeline
    device = -1
    if use_gpu_if_available and torch.cuda.is_available():
        device = 0
        print("[model_loader] GPU detected — pipeline will use device 0")
    else:
        print("[model_loader] No GPU detected or GPU usage disabled — using CPU")

    textgen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    print("[model_loader] Pipeline ready.")
    return textgen, tokenizer

