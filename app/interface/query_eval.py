import argparse
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import re
import tiktoken
import difflib

from query_plus import ask_question


def load_queries(file_path):
    ext = Path(file_path).suffix
    if ext == ".jsonl":
        with open(file_path, "r") as f:
            return [json.loads(line) for line in f]
    elif ext == ".json":
        with open(file_path, "r") as f:
            return json.load(f)
    else:
        raise ValueError("Only .json or .jsonl supported")


def evaluate_response(response, expected_answer, rules=None):
    result = {"passed": True, "rules": []}
    for rule in (rules or []):
        rule_type = rule["type"]
        rule_val = rule["value"]
        if rule_type == "contains":
            ok = rule_val.lower() in response.lower()
        elif rule_type == "not_contains":
            ok = rule_val.lower() not in response.lower()
        elif rule_type == "length_gt":
            ok = len(response.strip()) > int(rule_val)
        elif rule_type == "regex":
            ok = re.search(rule_val, response) is not None
        elif rule_type == "starts_with":
            ok = response.strip().lower().startswith(rule_val.lower())
        elif rule_type == "ends_with":
            ok = response.strip().lower().endswith(rule_val.lower())
        elif rule_type == "tokens_gt":
            enc = tiktoken.get_encoding("cl100k_base")
            ok = len(enc.encode(response)) > int(rule_val)
        elif rule_type == "fuzzy_match":
            response_clean = response.strip().lower()
            rule_val_clean = rule_val.strip().lower()
            window = response_clean[:len(rule_val_clean) + 20]
            ratio = difflib.SequenceMatcher(None, window, rule_val_clean).ratio()
            ok = ratio >= float(rule.get("threshold", 0.8))
        else:
            ok = True
        result["rules"].append({"rule": rule, "passed": ok})
        if not ok:
            result["passed"] = False
    return result


def run_batch_eval(input_file, model="mixtral", cot=False, out_file=None):
    queries = load_queries(input_file)
    results = []

    for entry in tqdm(queries, desc="Evaluating prompts"):
        q = entry["question"]
        if cot:
            q = "Let's think step by step. " + q

        print("\n---\nQ:", q)
        try:
            response_obj = ask_question(
                question=q,
                model_choice=model,
                filter_tag=entry.get("filter_tag"),
                filter_filename=entry.get("filter_file"),
                filter_all=entry.get("filter_all", False),
                stream=False,
                rules=entry.get("rules")
            )
            response = response_obj["answer"] if isinstance(response_obj, dict) else response_obj
            print(f"\n📤 Model Response:\n{response}")
            eval_result = evaluate_response(response, entry.get("expected_answer"), entry.get("rules"))
            print("\n📊 Rule Evaluation Details:")
            for rule_result in eval_result.get("rules", []):
                rule = rule_result["rule"]
                passed = rule_result["passed"]
                print(f"{'✔️ PASSED' if passed else '❌ FAILED'} — Rule: {rule}")
            print(f"\n📊 Evaluation Result: {'PASSED' if eval_result['passed'] else 'FAILED'}")
        except Exception as e:
            response = str(e)
            print(f"\n⚠️ Error during question execution: {response}")
            eval_result = {"passed": False, "error": str(e)}

        results.append({
            "question": q,
            "model": model,
            "response": response,
            "evaluation": eval_result
        })

    if out_file:
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results written to: {out_file}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch query evaluator with CoT and rule-based scoring")
    parser.add_argument("--input", required=True, help="Path to .json or .jsonl test cases")
    parser.add_argument("--model", choices=["mixtral", "llama3", "gpt4o"], default="mixtral")
    parser.add_argument("--cot", action="store_true", help="Enable chain-of-thought prefix")
    parser.add_argument("--output", help="Save full results to JSON file")
    args = parser.parse_args()

    run_batch_eval(
        input_file=args.input,
        model=args.model,
        cot=args.cot,
        out_file=args.output
    )
