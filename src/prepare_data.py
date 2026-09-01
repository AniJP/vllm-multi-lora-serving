"""Build instruction-tuning datasets for three LoRA tasks:
text-to-SQL, dialogue summarization, and named-entity extraction to JSON.
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset

SYSTEM_PROMPTS = {
    "sql": "You are a SQL generator. Given a table schema and a question, output only the SQL query.",
    "summarize": "You are a concise dialogue summarizer. Summarize the conversation in 1-2 sentences.",
    "extract": (
        'You are a named-entity extractor. Output a JSON list of {"text": ..., "type": ...} '
        "objects for every PERSON, ORG, LOC, and MISC entity found."
    ),
}


def build_sql(n_train: int, n_test: int):
    ds = load_dataset("b-mc2/sql-create-context", split="train")
    ds = ds.shuffle(seed=0).select(range(n_train + n_test))
    examples = []
    for row in ds:
        user = f"### Table schema:\n{row['context']}\n\n### Question:\n{row['question']}"
        examples.append({"user": user, "assistant": row["answer"]})
    return examples[:n_train], examples[n_train:]


def build_summarize(n_train: int, n_test: int):
    ds = load_dataset("knkarthick/samsum", split="train")
    ds = ds.shuffle(seed=0).select(range(n_train + n_test))
    examples = []
    for row in ds:
        user = f"### Conversation:\n{row['dialogue']}"
        examples.append({"user": user, "assistant": row["summary"]})
    return examples[:n_train], examples[n_train:]


# conll2003 ner_tags scheme: 0=O, 1/2=B/I-PER, 3/4=B/I-ORG, 5/6=B/I-LOC, 7/8=B/I-MISC
CONLL_LABELS = {1: "PERSON", 2: "PERSON", 3: "ORG", 4: "ORG", 5: "LOC", 6: "LOC", 7: "MISC", 8: "MISC"}
CONLL_BEGIN_TAGS = {1, 3, 5, 7}


def _bio_to_entities(tokens, tags):
    entities = []
    current, current_type = [], None
    for tok, tag in zip(tokens, tags):
        label = CONLL_LABELS.get(tag)
        if label is None:  # O
            if current:
                entities.append({"text": " ".join(current), "type": current_type})
                current, current_type = [], None
            continue
        if tag in CONLL_BEGIN_TAGS or label != current_type:
            if current:
                entities.append({"text": " ".join(current), "type": current_type})
            current, current_type = [tok], label
        else:
            current.append(tok)
    if current:
        entities.append({"text": " ".join(current), "type": current_type})
    return entities


def build_extract(n_train: int, n_test: int):
    ds = load_dataset("eriktks/conll2003", split="train", trust_remote_code=True)
    ds = ds.shuffle(seed=0).select(range(n_train + n_test))
    examples = []
    for row in ds:
        text = " ".join(row["tokens"])
        entities = _bio_to_entities(row["tokens"], row["ner_tags"])
        user = f"### Text:\n{text}"
        examples.append({"user": user, "assistant": json.dumps(entities)})
    return examples[:n_train], examples[n_train:]


BUILDERS = {"sql": build_sql, "summarize": build_summarize, "extract": build_extract}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="data")
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=20)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for task, builder in BUILDERS.items():
        train, test = builder(args.n_train, args.n_test)
        for split_name, rows in [("train", train), ("test", test)]:
            path = out / f"{task}_{split_name}.jsonl"
            with open(path, "w") as f:
                for row in rows:
                    row["system"] = SYSTEM_PROMPTS[task]
                    f.write(json.dumps(row) + "\n")
            print(f"wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
