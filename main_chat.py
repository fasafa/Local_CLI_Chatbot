
import argparse
import sys
import textwrap
from model_loader import load_model_and_tokenizer
from chat_memory import ChatMemory


def generate_reply(textgen_pipeline, prompt: str, max_new_tokens=80, temperature=0.7, top_k=50):
    """
    Call pipeline with simple generation hyperparams and return cleaned text.
    """
    outputs = textgen_pipeline(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        return_full_text=True,
    )

    generated_text = outputs[0]["generated_text"]

    # Remove the prompt prefix to get only the model continuation
    if generated_text.startswith(prompt):
        continuation = generated_text[len(prompt):].strip()
    else:
        continuation = generated_text.strip()

    # Remove redundant speaker labels or repeated prompts
    for sep in ["\nUser:", "\nBot:", "\nAssistant:", "\nHuman:"]:
        if sep in continuation:
            continuation = continuation.split(sep)[0].strip()

    if not continuation:
        continuation = "Sorry, I don't have an answer right now."
    return continuation


def run_cli(model_name: str, max_turns: int, max_new_tokens: int):
    textgen, tokenizer = load_model_and_tokenizer(model_name)
    memory = ChatMemory(max_turns=max_turns)

    banner = f"""
    Simple Local Chatbot
    Model: {model_name}
    Memory window (exchanges): {max_turns}
    Type '/exit' to quit, '/clear' to clear short-term memory.
    """
    print(textwrap.dedent(banner))

    system_prompt = (
        "You are a helpful, polite, and knowledgeable AI assistant. "
        "Answer user questions clearly, accurately, and conversationally.\n"
    )

    try:
        while True:
            user_input = input("User: ").strip()
            if not user_input:
                continue
            if user_input.lower() == "/exit":
                print("Exiting chatbot. Goodbye!")
                break
            if user_input.lower() == "/clear":
                memory.clear()
                print("[memory] cleared previous turns.")
                continue

            # Build prompt using memory + current input
            prompt = system_prompt + memory.get_context_prompt(user_input)

            # Generate reply
            bot_reply = generate_reply(
                textgen,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_k=50,
            )

            # Print and record
            print("Bot:", bot_reply)
            memory.add_turn(user_input, bot_reply)

    except KeyboardInterrupt:
        print("\nReceived KeyboardInterrupt. Exiting. Goodbye!")
        sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the local CLI chatbot.")
    parser.add_argument(
        "--model",
        type=str,
        default="distilgpt2",
        help="Hugging Face model id (default: distilgpt2). Use a small causal LM for CPU.",
    )
    parser.add_argument(
        "--memory",
        type=int,
        default=4,
        help="Number of recent exchanges to keep in sliding window (default: 4).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=80,
        help="Maximum tokens to generate per reply (default: 80).",
    )
    args = parser.parse_args()
    run_cli(model_name=args.model, max_turns=args.memory, max_new_tokens=args.max_new_tokens)
    
    
    
# python main_chat.py --model Qwen/Qwen2-0.5B --memory 4 --max-new-tokens 80

