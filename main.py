"""Main execution script for function calling tests."""

import json
from time import time
from typing import Any
from engine import FunctionCallingEngine
from llm_sdk import Small_LLM_Model


def main() -> None:
    """Run function selection and argument extraction against test prompts."""
    sdk = Small_LLM_Model(device="cpu")
    engine = FunctionCallingEngine(sdk=sdk)

    with open(
        "data/input/functions_definition.json", "r", encoding="utf-8"
    ) as f_def:
        functions_data: list[dict[str, Any]] = json.load(f_def)
    engine.add_functions(functions_data)

    with open(
        "data/input/function_calling_tests.json", "r", encoding="utf-8"
    ) as f_tests:
        prompts: list[dict[str, Any]] = json.load(f_tests)

    start = time()
    results: list[dict[str, Any]] = []

    for item in prompts:
        user_prompt: str = item["prompt"]
        print(f"\n--- {user_prompt}")

        func_name = engine.select_function(user_prompt)
        print(f"  Function: {func_name}")

        try:
            raw_args = engine.extract_arguments(user_prompt, func_name)
            args = engine.fix_argument_types(raw_args, func_name)
            print(f"  Args: {args}")
            results.append(
                {
                    "prompt": user_prompt,
                    "name": func_name,
                    "parameters": args,
                }
            )
        except json.JSONDecodeError as err:
            print(f"  Error parsing arguments: {err}")

    with open("data/output/outpout.json", "w", encoding="utf-8") as f_out:
        json.dump(results, f_out, indent=2)

    end = time()
    elapsed = end - start
    print(f"\nTotal time: {int(elapsed // 60)}mn {elapsed % 60:.1f}s")


if __name__ == "__main__":
    main()
