import argparse
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

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


def evaluate_response(response, expected, rules=None):
    result = {"passed": True, "rules": []}
    for rule in (rules or []):
        if rule["type"] == "contains":
            ok = rule["value"].lower() in response.lower()
        elif rule["type"] == "length_gt":
            ok = len(response.strip()) > int(rule["value"])
        elif rule["type"] == "regex":
            import re
            ok = re.search(rule["value"], response) is not None
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
            eval_result = evaluate_response(response, entry.get("expected"), entry.get("rules"))
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
