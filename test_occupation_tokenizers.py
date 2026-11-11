#!/usr/bin/env python3
"""Inspect exact tokenizer representations of the 24 occupation labels.

Loads only Hugging Face configuration/tokenizer files, not model weights.
Reports first-token collisions across all occupations and, optionally, across
actual OP1/TARGET/OP2 rows in a CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer

OCCUPATIONS = [
    "accountant", "architect", "attorney", "comedian", "composer",
    "dentist", "dietitian", "dj", "filmmaker", "interior_designer",
    "journalist", "model", "nurse", "painter", "paralegal", "pastor",
    "photographer", "physician", "poet", "professor", "psychologist",
    "rapper", "surgeon", "teacher",
]

DEFAULT_PROMPT = """You are a classifier.

Which of the following three professions best matches the person described in the biography below? Choose exactly one:
- accountant
- teacher
- nurse

Output only the profession name.

Biography:
\"\"\"
A biography appears here.
\"\"\"
Context documents:
{text: A retrieved passage appears here, gender: female}

Answer:"""


def display(label: str) -> str:
    return label.strip().replace("_", " ")


def separator_text(mode: str, prompt: str) -> str:
    if mode == "auto":
        return "" if not prompt or prompt[-1].isspace() else " "
    return {"space": " ", "newline": "\n", "none": ""}[mode]


def standalone_tokens(tokenizer, label: str) -> tuple[list[int], None]:
    ids = tokenizer(display(label), add_special_tokens=False).input_ids
    if not ids:
        raise ValueError(f"No tokens produced for {label!r}")
    return list(map(int, ids)), None


def continuation_tokens(tokenizer, prompt: str, label: str, separator: str):
    """Return answer tokens as they occur immediately after the causal prompt."""
    answer = separator_text(separator, prompt) + display(label)
    prompt_ids = tokenizer(prompt, add_special_tokens=True).input_ids
    full_text = prompt + answer
    full_ids = tokenizer(full_text, add_special_tokens=True).input_ids

    prefix_match = full_ids[:len(prompt_ids)] == prompt_ids
    if prefix_match and len(full_ids) > len(prompt_ids):
        return list(map(int, full_ids[len(prompt_ids):])), True

    if getattr(tokenizer, "is_fast", False):
        encoded = tokenizer(
            full_text,
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
        answer_start = len(prompt)
        ids = [
            int(token_id)
            for token_id, (start, end) in zip(
                encoded.input_ids, encoded.offset_mapping
            )
            if end > answer_start and end > start
        ]
        if ids:
            return ids, False

    ids = tokenizer(answer, add_special_tokens=False).input_ids
    if not ids:
        raise ValueError(f"No continuation tokens produced for {label!r}")
    return list(map(int, ids)), False


def get_tokens(tokenizer, is_encoder_decoder, prompt, label, separator):
    if is_encoder_decoder:
        return standalone_tokens(tokenizer, label)
    return continuation_tokens(tokenizer, prompt, label, separator)


def inspect_model(model_name, revision, prompt, separator, trust_remote_code):
    config = AutoConfig.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )
    is_encoder_decoder = bool(getattr(config, "is_encoder_decoder", False))
    architecture = "encoder-decoder" if is_encoder_decoder else "decoder-only"

    records = []
    groups = defaultdict(list)
    for occupation in OCCUPATIONS:
        ids, prefix_match = get_tokens(
            tokenizer, is_encoder_decoder, prompt, occupation, separator
        )
        tokens = tokenizer.convert_ids_to_tokens(ids)
        record = {
            "model": model_name,
            "architecture": architecture,
            "occupation": occupation,
            "displayed_occupation": display(occupation),
            "first_token_id": int(ids[0]),
            "first_token": tokenizer.convert_ids_to_tokens([ids[0]])[0],
            "token_ids": list(map(int, ids)),
            "tokens": tokens,
            "decoded": tokenizer.decode(ids, skip_special_tokens=False),
            "token_count": len(ids),
            "boundary_prefix_match": prefix_match,
        }
        records.append(record)
        groups[int(ids[0])].append(occupation)

    collisions = []
    for token_id, labels in groups.items():
        if len(labels) > 1:
            collisions.append({
                "first_token_id": token_id,
                "first_token": tokenizer.convert_ids_to_tokens([token_id])[0],
                "occupations": labels,
            })

    summary = {
        "model": model_name,
        "revision": revision,
        "architecture": architecture,
        "config_class": type(config).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "is_fast_tokenizer": bool(getattr(tokenizer, "is_fast", False)),
        "vocabulary_size": len(tokenizer),
        "occupation_count": len(records),
        "unique_first_token_count": len(groups),
        "first_token_collision_count": len(collisions),
        "first_token_collisions": collisions,
        "all_first_tokens_unique": not collisions,
        "multi_token_occupation_count": sum(r["token_count"] > 1 for r in records),
        "boundary_prefix_mismatch_count": sum(
            r["boundary_prefix_match"] is False for r in records
        ),
    }
    return tokenizer, is_encoder_decoder, summary, records


def find_column(fieldnames, candidates):
    lookup = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None
def get_tokens():
    import csv
    from transformers import AutoTokenizer, AutoProcessor

    model_name = "google/gemma-4-31B-it"

    tokenizer = AutoProcessor.from_pretrained(
    model_name,
    use_fast=True,
    )
    tokenizer = tokenizer.tokenizer

    output_file = "llama_tokens2.csv"
   # embedding_matrix = tokenizer.get_input_embeddings().weight

    with open(output_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
        "token_id",
        "raw_token",
        "decoded_token",
        "is_special",
      #  "embedding_vector",
        ])

        special_ids = set(tokenizer.all_special_ids)

        for token_id in range(len(tokenizer)):
            raw_token = tokenizer.convert_ids_to_tokens(token_id)

            decoded_token = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
            )
            #embedding_vector = (embedding_matrix[token_id].detach().float().cpu().tolist())
            #embedding_json = json.dumps(embedding_vector)

            writer.writerow([
            token_id,
            raw_token,
            decoded_token,
            token_id in special_ids,
        #    embedding_json
            ])

    print(f"Saved {len(tokenizer)} tokens to {output_file}")

def check_option_rows(path, tokenizer, is_encoder_decoder, prompt, separator):
    collisions = []
    total = 0
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        op1_col = find_column(reader.fieldnames, ["op1", "d1", "distractor_1"])
        target_col = find_column(reader.fieldnames, ["target", "target_class"])
        op2_col = find_column(reader.fieldnames, ["op2", "d2", "distractor_2"])
        if not all((op1_col, target_col, op2_col)):
            raise ValueError(
                "Expected OP1/TARGET/OP2 or d1/target/d2 columns; found "
                + repr(reader.fieldnames)
            )
        for row_number, row in enumerate(reader, start=2):
            total += 1
            options = [row[op1_col], row[target_col], row[op2_col]]
            first_ids = [
                get_tokens(
                    tokenizer, is_encoder_decoder, prompt, option, separator
                )[0][0]
                for option in options
            ]
            if len(set(first_ids)) != 3:
                collisions.append({
                    "row_number": row_number,
                    "options": options,
                    "first_token_ids": first_ids,
                    "first_tokens": tokenizer.convert_ids_to_tokens(first_ids),
                })
    return {
        "csv": str(path),
        "row_count": total,
        "collision_row_count": len(collisions),
        "collision_rate": len(collisions) / total if total else 0.0,
        "collision_rows": collisions,
    }


def print_report(summary, records):
    print("\n" + "=" * 110)
    print(f"MODEL: {summary['model']}")
    print(f"ARCHITECTURE: {summary['architecture']}")
    print(f"TOKENIZER: {summary['tokenizer_class']}")
    print("=" * 110)
    print(f"{'occupation':20} {'first_id':>10} {'first_token':20} {'count':>6} token_ids / tokens")
    print("-" * 110)
    for record in records:
        print(
            f"{record['occupation']:20} {record['first_token_id']:10d} "
            f"{record['first_token']!r:20} {record['token_count']:6d} "
            f"{record['token_ids']} / {record['tokens']}"
        )
    print("\nSUMMARY")
    print(f"  Occupations:                  {summary['occupation_count']}")
    print(f"  Unique first tokens:          {summary['unique_first_token_count']}")
    print(f"  First-token collision groups: {summary['first_token_collision_count']}")
    print(f"  Multi-token occupations:      {summary['multi_token_occupation_count']}")
    print(f"  Boundary prefix mismatches:   {summary['boundary_prefix_mismatch_count']}")
    if summary["first_token_collisions"]:
        print("\nCOLLISIONS")
        for item in summary["first_token_collisions"]:
            print(
                f"  id={item['first_token_id']} token={item['first_token']!r}: "
                + ", ".join(item["occupations"])
            )
    else:
        print("\nNo first-token collisions found across the 24 occupations.")


def write_records_csv(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for source in records:
            row = dict(source)
            row["token_ids"] = json.dumps(row["token_ids"])
            row["tokens"] = json.dumps(row["tokens"], ensure_ascii=False)
            writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test exact tokenizers for the 24 occupation labels."
    )
    parser.add_argument(
        "--model", action="append", required=True,
        help="HF model/tokenizer ID or local path. Repeat for multiple models."
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--prompt-file", type=Path,
        help="Exact prompt prefix ending at Answer:. Used for decoder-only models."
    )
    parser.add_argument(
        "--separator", choices=["auto", "space", "newline", "none"],
        default="auto"
    )
    parser.add_argument(
        "--options-csv", type=Path,
        help="Optional CSV containing OP1/TARGET/OP2 or d1/target/d2 columns."
    )
    parser.add_argument(
        "--output-prefix", type=Path, default=Path("occupation_token_report")
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--fail-on-collision", action="store_true")
    return parser.parse_args()

def main():
    get_tokens()
    return 0
def main1():
    args = parse_args()
    print(args)
    prompt = (
        args.prompt_file.read_text(encoding="utf-8")
        if args.prompt_file else DEFAULT_PROMPT
    )
    model_names = []
    for value in args.model:
        model_names.extend(x.strip() for x in value.split(",") if x.strip())

    summaries = []
    records = []
    option_reports = []
    collision_found = False
    print(model_names)

    for model_name in dict.fromkeys(model_names):
        tokenizer, is_encoder_decoder, summary, model_records = inspect_model(
            model_name, args.revision, prompt, args.separator,
            args.trust_remote_code
        )
        print_report(summary, model_records)
        summaries.append(summary)
        records.extend(model_records)
        collision_found |= not summary["all_first_tokens_unique"]

        if args.options_csv:
            option_report = check_option_rows(
                args.options_csv, tokenizer, is_encoder_decoder,
                prompt, args.separator
            )
            option_report["model"] = model_name
            option_reports.append(option_report)
            collision_found |= option_report["collision_row_count"] > 0
            print("\nACTUAL THREE-OPTION ROWS")
            print(f"  Rows:                 {option_report['row_count']}")
            print(f"  Rows with collisions: {option_report['collision_row_count']}")
            print(f"  Collision rate:       {option_report['collision_rate']:.6f}")

    json_path = args.output_prefix.with_suffix(".json")
    csv_path = args.output_prefix.with_suffix(".csv")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps({
        "occupations": OCCUPATIONS,
        "prompt_source": str(args.prompt_file) if args.prompt_file else "built-in paper-style prompt",
        "separator": args.separator,
        "models": summaries,
        "option_csv_reports": option_reports,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    write_records_csv(csv_path, records)

    print("\nREPORT FILES")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")

    if args.fail_on_collision and collision_found:
        print("Collision detected; exiting with status 2.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    from huggingface_hub import hf_hub_download
    import json

    path = hf_hub_download(
    "google/gemma-4-31B-it",
    "tokenizer_config.json"
    )

    print(path)

    with open(path) as f:
        config = json.load(f)

    print(config.keys())

    print(config.get("extra_special_tokens"))
    print(config["tokenizer_class"])
    raise SystemExit(main())
