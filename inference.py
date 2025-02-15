import argparse
import time

from dotenv import load_dotenv
from mlx_lm import load, stream_generate

load_dotenv(override=True)


def stream_inference(
    model_id: str,
    prompt: str,
    max_tokens: int,
    verbose: bool
):
    start_load_time = time.perf_counter()
    model, tokenizer = load(model_id)
    end_load_time = time.perf_counter()

    if verbose:
        print(
            f"Model loaded in {end_load_time - start_load_time :.2f} seconds"
        )

    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )

    output: str = ""
    start_inference_time = time.perf_counter()
    for response in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens):
        per_token_time = time.perf_counter()

        if verbose:
            print(
                f"\nPer Token Inference Time: {per_token_time - start_inference_time :.2f} seconds"
            )

        if response.finish_reason == "stop" and verbose:
            prompt_tokens = response.prompt_tokens
            prompt_tps = response.prompt_tps
            generation_tokens = response.generation_tokens
            generation_tps = response.generation_tps
            peak_memory = response.peak_memory
            total_generation_time = round(
                generation_tokens / generation_tps, 3)

            print("\n")
            print(f"Total generation time: {total_generation_time :.2f} s")
            print(f"Prompt tokens: {prompt_tokens} tokens")
            print(f"Prompt tps: {prompt_tps :.2f} tokens/s")
            print(f"Generation tokens: {generation_tokens} tokens")
            print(f"Generation tps: {generation_tps :.2f} tokens/s")
            print(f"Peak memory: {peak_memory :.2f} GB")

        if response.text == "<end_of_turn>":
            continue

        output += response.text
        print(response.text, end="", flush=True)

    print(f"\n\nOutput: {output}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM model inference on Apple Silicon Mac using the Apple MLX Framework."
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        help="Path to the model",
        required=True,
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Prompt for the LLM model",
        required=True,
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

    if verbose:
        print(f"Model: {model_id}")
        print(f"Prompt: {prompt}")
        print(f"Max tokens: {max_tokens}")
        print(f"Verbose mode: {verbose}")
        print()

    stream_inference(model_id, prompt, max_tokens, verbose)
