"""Serve a base model with 3 LoRA adapters loaded simultaneously via vLLM,
and demonstrate that routing the same engine to different adapters produces
task-specialized outputs from one running instance.

    python src/serve_demo.py --mode compare
    python src/serve_demo.py --mode interactive
"""

import argparse
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

TEST_PROMPTS = {
    "sql": (
        "You are a SQL generator. Given a table schema and a question, output only the SQL query.",
        "### Table schema:\nCREATE TABLE employees (id INT, name TEXT, department TEXT, salary INT)\n\n"
        "### Question:\nWhat is the average salary per department?",
    ),
    "summarize": (
        "You are a concise dialogue summarizer. Summarize the conversation in 1-2 sentences.",
        "### Conversation:\nAmy: Are we still on for the 3pm meeting?\nJake: Yeah, but can we push it to 3:30?\n"
        "Amy: Sure, I'll update the calendar invite.\nJake: Thanks, see you then.",
    ),
    "extract": (
        'You are a named-entity extractor. Output a JSON list of {"text": ..., "type": ...} objects '
        "for every PERSON, ORG, LOC, and MISC entity found.",
        "### Text:\nSatya Nadella, CEO of Microsoft, met with officials in Berlin last week.",
    ),
}

TASKS = list(TEST_PROMPTS)


def build_prompt(tokenizer, system, user):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--adapters-dir", default="adapters")
    ap.add_argument("--mode", choices=["compare", "interactive"], default="compare")
    ap.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help=(
            "Llama-3.2's default 131072 context needs ~14GiB of KV cache just to "
            "start, which doesn't fit a 16GB GPU alongside the model weights. Our "
            "prompts + 256-token generations don't need anywhere near that."
        ),
    )
    args = ap.parse_args()

    adapter_paths = {t: str(Path(args.adapters_dir) / f"{t}-lora") for t in TASKS}

    llm = LLM(
        model=args.base_model,
        enable_lora=True,
        max_loras=len(TASKS),
        max_lora_rank=16,
        max_model_len=args.max_model_len,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=256)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    lora_requests = {t: LoRARequest(f"{t}-adapter", i + 1, adapter_paths[t]) for i, t in enumerate(TASKS)}

    if args.mode == "compare":
        for task in TASKS:
            system, user = TEST_PROMPTS[task]
            prompt = build_prompt(tokenizer, system, user)

            base_out = llm.generate([prompt], sampling)[0].outputs[0].text
            lora_out = llm.generate([prompt], sampling, lora_request=lora_requests[task])[0].outputs[0].text

            print(f"\n{'=' * 60}\nTASK: {task}\n{'=' * 60}")
            print(f"PROMPT:\n{user}\n")
            print(f"BASE MODEL (no adapter):\n{base_out.strip()}\n")
            print(f"{task.upper()}-LORA ADAPTER:\n{lora_out.strip()}\n")
        return

    while True:
        task = input(f"\nadapter [{'/'.join(TASKS)}/base/quit]: ").strip()
        if task == "quit":
            break
        user = input("prompt: ").strip()
        system = TEST_PROMPTS.get(task, ("You are a helpful assistant.", ""))[0]
        prompt = build_prompt(tokenizer, system, user)
        req = lora_requests.get(task)
        out = llm.generate([prompt], sampling, lora_request=req)[0].outputs[0].text
        print(f"\n{out.strip()}")


if __name__ == "__main__":
    main()
