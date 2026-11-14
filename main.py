import json
from llm_sdk import Small_LLM_Model
import numpy as np
from time import time


sdk = Small_LLM_Model(device="cpu")

prompt_select_tpl = open("data/prompt_select_function.txt").read()
prompt_args_tpl = open("data/prompt_extract_args.txt").read()


def select_function(user_prompt: str, functions: list[dict]) -> str:
    """
    Use the LLM to select the best function for a user request.
    Trie-based early exit: stops generating tokens as soon as the
    generated prefix uniquely identifies one function name.
    e.g. after generating "fn_add" we know it can only be "fn_add_numbers" → stop.
    """
    func_names = [f["name"] for f in functions]

    func_list = "\n".join(
        f'- {f["name"]}: {f["description"]}' for f in functions
    )

    prompt = (prompt_select_tpl
              .replace("{functions_list}", func_list)
              .replace("{user_prompt}", user_prompt))

    ids = sdk.encode(prompt).tolist()[0]
    generated_ids = []

    for _ in range(50):
        logits = sdk.get_logits_from_input_ids(ids)
        token_id = int(np.argmax(logits))
        ids.append(token_id)
        generated_ids.append(token_id)

        generated_text = sdk.decode(generated_ids).strip().strip('"')
        if not generated_text:
            continue

        candidates = [n for n in func_names if n.startswith(generated_text)]

        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) == 0:
            for n in func_names:
                if generated_text.startswith(n):
                    return n
            break

    generated_text = sdk.decode(generated_ids).strip().strip('"')
    best, best_len = func_names[0], 0
    for n in func_names:
        match_len = 0
        for a, b in zip(generated_text, n):
            if a == b:
                match_len += 1
            else:
                break
        if match_len > best_len:
            best, best_len = n, match_len
    return best


def extract_arguments(user_prompt: str, functions: list[dict], function_name: str) -> dict:
    """
    Use the LLM to extract arguments from the user request for the chosen function.
    Only generates the JSON arguments object (not the full output wrapper).
    Uses brace tracking to stop as soon as the JSON object is complete.
    """
    func_def = next(f for f in functions if f["name"] == function_name)
    params = func_def["parameters"]

    param_desc = ", ".join(
        f'{name}: {info["type"]}' for name, info in params.items()
    )

    prompt = (prompt_args_tpl
              .replace("{function_name}", function_name)
              .replace("{param_desc}", param_desc)
              .replace("{user_prompt}", user_prompt))

    ids = sdk.encode(prompt).tolist()[0]
    response_ids = []
    br = ["{"]

    for _ in range(200):
        logits = sdk.get_logits_from_input_ids(ids)
        token_id = int(np.argmax(logits))
        ids.append(token_id)
        response_ids.append(token_id)

        t = sdk.decode(token_id)
        if "{" in t:
            br.extend(["{"] * t.count("{"))
        if "}" in t:
            for _ in range(t.count("}")):
                if br:
                    br.pop()
        if not br:
            text = "{" + sdk.decode(response_ids)
            return json.loads(text)

    text = "{" + sdk.decode(response_ids)
    if not text.rstrip().endswith("}"):
        text += "}"
    return json.loads(text)


functions = json.load(open("data/input/functions_definition.json"))
prompts = json.load(open("data/input/function_calling_tests.json"))

start = time()
results = []
for item in prompts:
    user_prompt = item["prompt"]
    print(f"\n--- {user_prompt}")

    func_name = select_function(user_prompt, functions)
    print(f"  Function: {func_name}")

    try:
        args = extract_arguments(user_prompt, functions, func_name)
        print(f"  Args: {args}")
        results.append({
            "prompt": user_prompt,
            "name": func_name,
            "parameters": args
        })
    except json.JSONDecodeError as e:
        print(f"  Error parsing arguments: {e}")

json.dump(results, open("data/output/outpout.json", "w"), indent=2)
end = time()
print(f"\nTotal time: {int((end - start) // 60)}mn {(end - start) % 60:.1f}s")
