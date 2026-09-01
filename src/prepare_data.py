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


# eriktks/conll2003 ships a legacy Python loading script, which recent
# `datasets` versions refuse to run. Try Parquet-backed sources instead, in
# order, and read the BIO label names off the dataset's own feature schema
# rather than hardcoding tag ids (which differ between repackagings).
NER_CANDIDATES = [
    {"path": "eriktks/conll2003", "kwargs": {"revision": "refs/convert/parquet"}, "tag_col": "ner_tags"},
    {"path": "tner/conll2003", "kwargs": {}, "tag_col": "tags"},
]

# Some repackagings abbreviate PER; keep output wording consistent with
# SYSTEM_PROMPTS["extract"] either way.
LABEL_DISPLAY = {"PER": "PERSON"}


def _load_ner_dataset():
    last_err = None
    for cand in NER_CANDIDATES:
        try:
            ds = load_dataset(cand["path"], split="train", **cand["kwargs"])
            return ds, cand["tag_col"]
        except Exception as e:  # noqa: BLE001 - trying multiple sources on purpose
            last_err = e
    raise RuntimeError(
        "Could not load a CoNLL-2003-style NER dataset from any known source "
        f"({[c['path'] for c in NER_CANDIDATES]}). Hugging Face deprecated "
        "script-based dataset loading, so a mirror may have moved again - "
        "pick a current Parquet-backed NER dataset and update NER_CANDIDATES."
    ) from last_err


def _bio_to_entities(tokens, tag_ids, id2label):
    entities = []
    current, current_type = [], None
    for tok, tag_id in zip(tokens, tag_ids):
        label = id2label[tag_id]
        if label == "O":
            if current:
                entities.append({"text": " ".join(current), "type": current_type})
                current, current_type = [], None
            continue
        prefix, ent_type = label.split("-", 1)
        ent_type = LABEL_DISPLAY.get(ent_type, ent_type)
        if prefix == "B" or ent_type != current_type:
            if current:
                entities.append({"text": " ".join(current), "type": current_type})
            current, current_type = [tok], ent_type
        else:
            current.append(tok)
    if current:
        entities.append({"text": " ".join(current), "type": current_type})
    return entities


def build_extract(n_train: int, n_test: int):
    ds, tag_col = _load_ner_dataset()
    id2label = ds.features[tag_col].feature.names
    ds = ds.shuffle(seed=0).select(range(n_train + n_test))
    examples = []
    for row in ds:
        text = " ".join(row["tokens"])
        entities = _bio_to_entities(row["tokens"], row[tag_col], id2label)
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
