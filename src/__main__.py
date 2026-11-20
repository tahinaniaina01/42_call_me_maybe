"""Main execution script for function calling tests."""

import json
import sys
from time import time
from typing import Any, Sequence

from .engine import FunctionCallingEngine
from .parse import Parser
from llm_sdk import Small_LLM_Model

def main(argv: Sequence[str] | None = None) -> int:
    """Run function selection and argument extraction against test prompts."""
    args = Parser.parse(argv)

    try:
        functions_data = Parser.validate_functions(
            Parser.load_json(
                args.functions_definition, "functions_definition"
            )
        )
        prompts = Parser.validate_prompts(
            Parser.load_json(args.input, "input")
        )

        sdk = Small_LLM_Model(device="cpu")
        engine = FunctionCallingEngine(sdk=sdk)
        engine.add_functions(functions_data)

        start = time()
        results: list[dict[str, Any]] = []

        for index, item in enumerate(prompts):
            user_prompt = item["prompt"]
            print(f"\n--- {user_prompt}")

            try:
                func_name = engine.select_function(user_prompt)
                print(f"  Function: {func_name}")

                raw_args = engine.extract_arguments(user_prompt, func_name)
                args_fixed = engine.fix_argument_types(raw_args, func_name)
                args_fixed = Parser.validate_extracted_arguments(
                    args_fixed, func_name, functions_data
                )
                print(f"  Args: {args_fixed}")
                results.append(
                    {
                        "prompt": user_prompt,
                        "name": func_name,
                        "parameters": args_fixed,
                    }
                )
            except (json.JSONDecodeError, KeyError, StopIteration, ValueError) as error:
                print(
                    f"  Error processing input[{index}] "
                    f"('{user_prompt}'): {error}",
                    file=sys.stderr,
                )
            except Exception as error:
                print(
                    f"  Unexpected error processing input[{index}] "
                    f"('{user_prompt}'): {type(error).__name__}: {error}",
                    file=sys.stderr,
                )

        Parser.write_results(args.output, results)

        end = time()
        elapsed = end - start
        print(f"\nTotal time: {int(elapsed // 60)}mn {elapsed % 60:.1f}s")
        return 0
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Unexpected error: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
