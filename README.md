# vllm-multi-lora-serving

Train 3 task-specialized **LoRA adapters** on one base model, then serve all
of them **simultaneously from a single vLLM engine instance** — no reloading,
no separate model copies in memory. Each request picks which adapter to route
to at inference time via vLLM's `LoRARequest` API.

## Why this project

vLLM supports loading multiple LoRA adapters on top of one base model and
routing individual requests to different adapters on the fly
(`enable_lora=True`, `max_loras`, `LoRARequest`). This is a genuinely useful
production pattern — e.g. one base model serving several fine-tuned "skills"
— but rarely gets demonstrated end-to-end. This project trains the adapters
itself (not just plugging in pre-trained ones) and proves the multi-adapter
serving actually works by showing outputs diverge correctly per task from one
running engine.

## The 3 adapters

| Task | Adapter | Dataset | What it does |
|------|---------|---------|---------------|
| `sql` | text-to-SQL | [`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context) | table schema + question → SQL query |
| `summarize` | dialogue summarization | [`knkarthick/samsum`](https://huggingface.co/datasets/knkarthick/samsum) | conversation → 1-2 sentence summary |
| `extract` | NER → JSON extraction | [`eriktks/conll2003`](https://huggingface.co/datasets/eriktks/conll2003) | free text → JSON list of `{text, type}` entities |

Base model: **`meta-llama/Llama-3.2-3B-Instruct`** (gated — see setup below).
3B keeps training + multi-adapter serving comfortable on a single 16GB GPU
(e.g. Kaggle's free T4).

## Project layout

```
src/
  prepare_data.py   # downloads + formats all 3 datasets into instruction pairs (JSONL)
  train_lora.py      # generic LoRA trainer, run once per --task
  serve_demo.py       # loads base model + all 3 adapters into one vLLM engine, compares outputs
notebooks/
  kaggle_train_and_serve.ipynb   # end-to-end Kaggle notebook wiring the above together
```

## Setup

`meta-llama/Llama-3.2-3B-Instruct` is gated on Hugging Face:

1. Accept the license at https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
2. Create a read-access token at https://huggingface.co/settings/tokens
3. On Kaggle: Add-ons → Secrets → add it as `HF_TOKEN`, and in the notebook:
   ```python
   from huggingface_hub import login
   from kaggle_secrets import UserSecretsClient
   login(UserSecretsClient().get_secret("HF_TOKEN"))
   ```

If you'd rather skip the gating step entirely, swap `--base-model` to an
ungated equivalent like `Qwen/Qwen2.5-3B-Instruct` everywhere below — the
scripts don't assume any Llama-specific behavior.

## Running it (locally or on any single-GPU machine)

```bash
pip install -r requirements.txt

python src/prepare_data.py --output-dir data

python src/train_lora.py --task sql        --data-dir data --output-dir adapters
python src/train_lora.py --task summarize  --data-dir data --output-dir adapters
python src/train_lora.py --task extract    --data-dir data --output-dir adapters

pip install vllm   # separate install — vLLM pins its own torch/cuda build
python src/serve_demo.py --mode compare
```

`--mode compare` runs one held-out prompt per task through the **base model**
(no adapter) and through **its matching LoRA adapter**, printing both so the
specialization is visible. `--mode interactive` lets you pick an adapter and
type your own prompts against the running engine.

You can also stand up vLLM's OpenAI-compatible server with all 3 adapters
hot-loaded and hit it with curl / any OpenAI client:

```bash
vllm serve meta-llama/Llama-3.2-3B-Instruct \
  --enable-lora \
  --lora-modules sql-adapter=adapters/sql-lora summarize-adapter=adapters/summarize-lora extract-adapter=adapters/extract-lora

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "sql-adapter", "messages": [{"role": "user", "content": "..."}]}'
```

## Running on Kaggle

See `notebooks/kaggle_train_and_serve.ipynb`. It clones this repo (push it to
GitHub first), installs dependencies, runs data prep + training for all 3
tasks, then installs vLLM and runs the comparison demo — all within one
session. Training each adapter on ~2,000 examples for 2 epochs on a T4 takes
roughly 20-40 minutes per task depending on sequence length.

## Notes

- 4-bit quantization (`bitsandbytes`, NF4) is used during LoRA fine-tuning to
  keep memory low; vLLM serves the adapters against the full-precision base
  model at inference time.
- LoRA rank 16 / alpha 32 on all attention + MLP projection matrices — a
  reasonable default for task-specialization on a 3B model with ~2k examples
  per task; increase rank or examples if outputs look under-fit.
- This project is deliberately **not** a benchmarking exercise — no
  throughput/latency numbers are collected. The deliverable is the working
  multi-adapter serving demo and the trained adapters themselves.
