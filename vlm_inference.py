import argparse
import os
from pathlib import Path
import time
from typing import List, Union

import mlx.core as mx
from dotenv import load_dotenv
from mlx import nn
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config, stream_generate
from mlx_vlm.video_generate import process_vision_info
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

load_dotenv(override=True)


def load_model(model_id: str, verbose: bool):
    start = time.perf_counter()
    model, processor = load(model_id)
    config = load_config(model_id)
    end = time.perf_counter()

    if verbose:
        print(f"Model loaded in {end - start :.2f} seconds")

    return model, processor, config


def process_prompt(
    task: str,
    processor: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    config: dict,
    images: List[str],
    videos: List[str],
    prompt: str,
):
    match task:
        case "image":
            messages = apply_chat_template(
                processor,
                config,
                prompt,
                num_images=len(images)
            )

            return messages
        case "video":
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": videos[0],
                            "max_pixels": 360 * 360,
                            "fps": 1.0,
                        },
                        {
                            "type": "text",
                            "text": prompt
                        },
                    ],
                }
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            input_ids = mx.array(inputs['input_ids'])
            pixel_values = mx.array(inputs['pixel_values_videos'])
            mask = mx.array(inputs['attention_mask'])
            image_grid_thw = mx.array(inputs['video_grid_thw'])
            kwargs = {
                "image_grid_thw": image_grid_thw,
            }
            kwargs["video"] = videos[0]
            kwargs["input_ids"] = input_ids
            kwargs["pixel_values"] = pixel_values
            kwargs["mask"] = mask

            return kwargs


def image_stream_inference(
    model: nn.Module,
    processor: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    prompt: str,
    images: List[str],
    max_tokens: int,
    temperature: float = 0.0,
    verbose: bool = False,
):
    output: str = ""
    start_inference_time = time.perf_counter()
    for response in stream_generate(
        model,
        processor,
        prompt,
        images,
        temp=temperature,
        max_tokens=max_tokens
    ):
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
            print(
                f"Total generation time: {total_generation_time :.2f} s"
            )
            print(f"Prompt tokens: {prompt_tokens} tokens")
            print(f"Prompt tps: {prompt_tps :.2f} tokens/s")
            print(f"Generation tokens: {generation_tokens} tokens")
            print(f"Generation tps: {generation_tps :.2f} tokens/s")
            print(f"Peak memory: {peak_memory :.2f} GB")
            print(f"\n\nOutput: {output}")

        if response.text == "<end_of_turn>":
            continue

        output += response.text
        print(response.text, end="", flush=True)


def video_stream_inference(
    model: nn.Module,
    processor: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    prompt: str,
    max_tokens: int,
    temperature: float = 0.0,
    verbose: bool = False,
    **kwargs,
):
    output: str = ""
    start_inference_time = time.perf_counter()
    for response in stream_generate(
        model,
        processor,
        prompt,
        temp=temperature,
        max_tokens=max_tokens,
        **kwargs
    ):
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
            print(
                f"Total generation time: {total_generation_time :.2f} s"
            )
            print(f"Prompt tokens: {prompt_tokens} tokens")
            print(f"Prompt tps: {prompt_tps :.2f} tokens/s")
            print(f"Generation tokens: {generation_tokens} tokens")
            print(f"Generation tps: {generation_tps :.2f} tokens/s")
            print(f"Peak memory: {peak_memory :.2f} GB")
            print(f"\n\nOutput: {output}")

        if response.text == "<end_of_turn>":
            continue

        output += response.text
        print(response.text, end="", flush=True)


def stream_inference(
    task: str,
    model: nn.Module,
    processor: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    prompt: str,
    images: List[str],
    max_tokens: int,
    temperature: float,
    verbose: bool,
    **kwargs,
):
    match task:
        case "image":
            image_stream_inference(
                model,
                processor,
                prompt,
                images,
                max_tokens,
                temperature,
                verbose
            )
        case "video":
            video_stream_inference(
                model,
                processor,
                prompt,
                max_tokens,
                temperature,
                verbose,
                **kwargs
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="VLM model inference on Apple Silicon Mac using the Apple MLX Framework."
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["image", "video"],
        help="Task type, image or video",
        required=True,
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
        help="Prompt for the VLM model",
        required=True,
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        help="Path to the image directory",
        default="./images",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        help="Path to the video directory",
        default="./videos",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature for sampling",
        default=0.0,
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

    task: str = args.task
    model_id: str = args.model
    prompt: str = args.prompt
    image_dir: str = args.image_dir
    video_dir: str = args.video_dir
    max_tokens: int = args.max_tokens
    temperature: float = args.temperature
    verbose: bool = args.verbose

    if verbose:
        print(f"Task: {task}")
        print(f"Model: {model_id}")
        print(f"Prompt: {prompt}")
        print(
            f"Image Directory: {image_dir}, Exists: {os.path.exists(image_dir)}"
        )
        print(
            f"Video Directory: {video_dir}, Exists: {os.path.exists(video_dir)}"
        )
        print(f"Max Tokens: {max_tokens}")
        print(f"Temperature: {temperature}")
        print(f"Verbose: {verbose}")

    if not os.path.exists(image_dir):
        os.makedirs(image_dir, exist_ok=True)

    if not os.path.exists(video_dir):
        os.makedirs(video_dir, exist_ok=True)

    images = []
    videos = []

    match task:
        case "image":
            IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
            for ext in IMAGE_EXTENSIONS:
                images.extend(Path(image_dir).glob(f'*{ext}'))
                images.extend(Path(image_dir).glob(f'*{ext.upper()}'))

            if len(images) == 0:
                raise ValueError("No images found in the image directory!!!")
            elif len(images) > 1:
                print("[Warning]: Only the first image will be used for inference!!!")
        case "video":
            VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv'}
            for ext in VIDEO_EXTENSIONS:
                videos.extend(Path(image_dir).glob(f'*{ext}'))
                videos.extend(Path(image_dir).glob(f'*{ext.upper()}'))

            if len(videos) == 0:
                raise ValueError("No videos found in the video directory!!!")
            elif len(videos) > 1:
                print("[Warning]: Only the first video will be used for inference!!!")
        case _:
            raise ValueError(f"Task type {task} is not supported!!!")

    model, processor, config = load_model(model_id, verbose)
    input_ids = process_prompt(
        task,
        processor,
        config,
        images[:1],
        videos[:1],
        prompt
    )
    stream_inference(
        task,
        model,
        processor,
        prompt,
        images,
        max_tokens,
        temperature,
        verbose,
        **input_ids
    )
