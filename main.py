import json
from llm_sdk import Small_LLM_Model
import numpy as np
from time import time


sdk = Small_LLM_Model(device="cpu")

# Load prompt templates
prompt_select_tpl = open("data/prompt_select_function.txt").read()
prompt_args_tpl = open("data/prompt_extract_args.txt").read()


def select_function(user_prompt: str, functions: list[dict]) -> str:
    """
    Sélectionne la fonction de façon 100% dynamique par correspondance de préfixe.
    Dès que le texte généré correspond au préfixe d'UNE SEULE fonction parmi la liste,
    on s'arrête immédiatement et on retourne cette fonction.
    """
    func_names = [f["name"] for f in functions]
    func_list = "\n".join(f'- {f["name"]}: {f.get("description", "")}' for f in functions)

    prompt = (prompt_select_tpl
              .replace("{functions_list}", func_list)
              .replace("{user_prompt}", user_prompt))

    ids = sdk.encode(prompt).tolist()[0]
    generated_ids = []

    for _ in range(30):
        logits = sdk.get_logits_from_input_ids(ids)
        token_id = int(np.argmax(logits))
        ids.append(token_id)
        generated_ids.append(token_id)

        text = sdk.decode(generated_ids).strip().strip('"\'`')
        if not text:
            continue

        if text in func_names:
            return text

        candidates = [n for n in func_names if n.startswith(text)]

        if len(candidates) == 1:
            return candidates[0]

        if len(candidates) == 0:
            for n in func_names:
                if n in text or text.startswith(n):
                    return n
            break

    text = sdk.decode(generated_ids).strip().strip('"\'`')
    best_candidate = func_names[0]
    best_len = -1
    for n in func_names:
        match_len = 0
        for c1, c2 in zip(text, n):
            if c1 == c2:
                match_len += 1
            else:
                break
        if match_len > best_len:
            best_len = match_len
            best_candidate = n

    return best_candidate


def extract_arguments(user_prompt: str, functions: list[dict], function_name: str) -> dict:
    """
    Extrait les arguments au format JSON pour la fonction sélectionnée.
    Le prompt montre explicitement les noms des paramètres attendus et le format JSON.
    """
    func_def = next(f for f in functions if f["name"] == function_name)
    params = func_def.get("parameters", {})
    if not params:
        return {}

    param_list = "\n".join(f'- {name} ({info.get("type", "any")})' for name, info in params.items())
    format_items = ", ".join(f'"{name}": <{info.get("type", "value")}>' for name, info in params.items())
    param_format = "{" + format_items + "}"

    prompt = (prompt_args_tpl
              .replace("{function_name}", function_name)
              .replace("{param_list}", param_list)
              .replace("{param_format}", param_format)
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
            text = '{"' + sdk.decode(response_ids)
            return json.loads(text)

    text = '{"' + sdk.decode(response_ids)
    if not text.rstrip().endswith("}"):
        text += "}"
    return json.loads(text)


def fix_argument_types(args: dict, func_def: dict) -> dict:
    """
    Ne conserve STRICTEMENT que les paramètres déclarés dans la fonction,
    et convertit selon les types :
    - integer, int -> entier
    - number, float -> flottant
    - string, str -> texte
    - boolean, bool -> booléen
    """
    params = func_def.get("parameters", {})
    fixed = {}
    for k, info in params.items():
        if k in args:
            v = args[k]
            t = str(info.get("type", "")).lower()
            if t in ("integer", "int") and isinstance(v, (int, float)):
                fixed[k] = int(v)
            elif t in ("number", "float") and isinstance(v, (int, float)):
                fixed[k] = float(v)
            elif t in ("string", "str") and v is not None:
                fixed[k] = str(v)
            elif t in ("boolean", "bool") and isinstance(v, (bool, int, str)):
                fixed[k] = v.lower() == "true" if isinstance(v, str) else bool(v)
            else:
                fixed[k] = v
    return fixed


# --- Main ---
functions = json.load(open("data/test/functions.json"))
prompts = json.load(open("data/test/prompts.json"))

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
print(f"\nTotal time: {int((end - start) // 60)}mn {(end - start) % 60:.1f}s")
