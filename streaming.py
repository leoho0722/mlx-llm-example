import argparse

from mlx_lm import load, stream_generate


def streaming_inference(model_id: str, prompt: str, max_tokens: int, verbose: bool):
    if verbose:
        print(f"Model: {model_id}")
        print(f"Prompt: {prompt}")
        print(f"Max tokens: {max_tokens}")
        print(f"Verbose mode: {verbose}")
        print()

    model, tokenizer = load(model_id)
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )

    for response in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens):
        if response.finish_reason == "stop":
            if verbose:
                prompt_tokens = response.prompt_tokens
                prompt_tps = response.prompt_tps
                generation_tokens = response.generation_tokens
                generation_tps = response.generation_tps
                peak_memory = response.peak_memory

                print("\n")
                print(f"Prompt tokens: {prompt_tokens} tokens")
                print(f"Prompt tps: {prompt_tps :.2f} tokens/s")
                print(f"Generation tokens: {generation_tokens} tokens")
                print(f"Generation tps: {generation_tps :.2f} tokens/s")
                print(f"Peak memory: {peak_memory :.2f} GB")

        if response.text == "<end_of_turn>":
            continue
        print(response.text, end="", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM model streaming inference using the Apple MLX Framework."
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        help="Model ID",
        default="mlx-community/gemma-2-9b-it-4bit",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Prompt to generate text from",
        default="What is the largest country in the world?",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        help="Maximum tokens to generate",
        default=512,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose mode"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model_id: str = args.model
    prompt: str = args.prompt
    max_tokens: int = args.max_tokens
    verbose: bool = args.verbose

    streaming_inference(model_id, prompt, max_tokens, verbose)
