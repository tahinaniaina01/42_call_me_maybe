import json
from llm_sdk import Small_LLM_Model
import numpy as np


sdk = Small_LLM_Model()

def generate(prompt="Bonjour", max_token=500):
    ids = sdk.encode(prompt).tolist()[0]
    response = []
    br = []
    for _ in range(max_token):
        logits = sdk.get_logits_from_input_ids(ids)
        am = np.argmax(logits)
        ids.append(am)
        response.append(am)
        t = sdk.decode(am)
        if "{" in t:
            br.append(["{" for _ in range(t.count("{"))])
        if "}" in t:
            if br:
                for _ in range(t.count("}")):
                    br.pop(-1)
        if len(br) == 0:
            return sdk.decode(response)
        print(t)
    return sdk.decode(response)

template = open("data/prompt_system.txt").read()
functions = json.load(open("data/test/functions.json"))
prompts = json.load(open("data/test/prompts.json"))

results = []
for item in prompts:
    p = (template
         .replace("{functions_json}", json.dumps(functions, indent=2))
         .replace("{user_prompt}", item["prompt"]))
    raw = generate(prompt=p)
    # print(raw)
    try:
        results.append(json.loads(raw))
    except json.JSONDecodeError:
        print(f"Error parsing: {raw}")

json.dump(results, open("data/output/outpout.json", "w"), indent=2)
