import json
from llm_sdk import Small_LLM_Model
import numpy as np
from time import time


sdk = Small_LLM_Model(device="cpu")

prompt_select_tpl = open("data/prompt_select_function.txt").read()
prompt_args_tpl = open("data/prompt_extract_args.txt").read()


def select_function(user_prompt: str, functions: list[dict]) -> str:
    func_names = [f["name"] for f in functions]
    func_list = "\n".join(f'- {f["name"]}: {f["description"]}' for f in functions)

    prompt = (prompt_select_tpl
              .replace("{functions_list}", func_list)
              .replace("{user_prompt}", user_prompt))

    ids = sdk.encode(prompt).tolist()[0]
    
    next_tokens = {}
    for name in func_names:
        toks = sdk.encode(name).tolist()[0]
        if len(toks) > 1:
            next_tokens[toks[1]] = name

    logits = sdk.get_logits_from_input_ids(ids)
    
    scores = {name: logits[tok_id] for tok_id, name in next_tokens.items()}
    best_candidate = max(scores, key=scores.get)
    return best_candidate


def extract_arguments(user_prompt: str, functions: list[dict], function_name: str) -> dict:
    func_def = next(f for f in functions if f["name"] == function_name)
    params = func_def.get("parameters", {})
    if not params:
        return {}

    param_desc = ", ".join(f'{name}: {info["type"]}' for name, info in params.items())

    prompt = (prompt_args_tpl
              .replace("{function_name}", function_name)
              .replace("{param_desc}", param_desc)
              .replace("{user_prompt}", user_prompt))

    ids = sdk.encode(prompt).tolist()[0]
    response_ids = []
    br = ["{"]

    for _ in range(80):
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


def fix_argument_types(args: dict, func_def: dict) -> dict:
    params = func_def.get("parameters", {})
    fixed = {}
    for k, v in args.items():
        if k in params:
            expected_type = params[k].get("type")
            if expected_type == "number" and isinstance(v, (int, float)):
                fixed[k] = float(v)
            elif expected_type == "integer" and isinstance(v, (int, float)):
                fixed[k] = int(v)
            else:
                fixed[k] = v
        else:
            fixed[k] = v
    return fixed


functions = json.load(open("data/input/functions_definition.json"))
prompts = json.load(open("data/input/function_calling_tests.json"))

start = time()
results = []
for item in prompts:
    user_prompt = item["prompt"]
    print(f"\n--- {user_prompt}")

    func_name = select_function(user_prompt, functions)
    print(f"  Function: {func_name}")

    func_def = next(f for f in functions if f["name"] == func_name)
    try:
        args = extract_arguments(user_prompt, functions, func_name)
        args = fix_argument_types(args, func_def)
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
print(f"\nTotal time: {int((end - start) // 60)}mn {(end - start) % 60:02f}s")
