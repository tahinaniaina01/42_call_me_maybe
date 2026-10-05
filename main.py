import json
from llm_sdk import Small_LLM_Model
import numpy as np
from time import time

sdk = Small_LLM_Model()

def generate(prompt="", max_token=500, user_prompt="Bonjour"):
    ids = sdk.encode(prompt).tolist()[0]
    a = '{"prompt": "' + user_prompt + '", "name": "'
    response = sdk.encode(a).tolist()[0]
    br = ["{"]
    for _ in range(max_token):
        logits = sdk.get_logits_from_input_ids(ids)
        am = np.argmax(logits)
        ids.append(am)
        response.append(am)
        t = sdk.decode(am)
        print(f"t: [{t}]")
        if '",' in t:
            new = sdk.encode('"parameters": {').tolist()[0]
            br.append("{")
            ids.extend(new)
            response.extend(new)
        if "{" in t:
            br.extend(["{" for _ in range(t.count("{"))])
        if "}" in t:
            if br:
                for _ in range(t.count("}")):
                    br.pop(-1)
        if not br:
            return sdk.decode(response)
    return sdk.decode(response)


template = open("data/prompt_system.txt").read()
functions = json.load(open("data/test/functions.json"))
prompts = json.load(open("data/test/prompts.json"))

start = time()
results = []
for item in prompts:
    p = (template
         .replace("{functions_json}", json.dumps(functions, indent=2))
         .replace("{user_prompt}", item["prompt"]))
    raw = generate(prompt=p, user_prompt=item["prompt"])
    # print(raw)
    try:
        results.append(json.loads(raw))
    except json.JSONDecodeError:
        print(f"Error parsing: {raw}")


json.dump(results, open("data/output/outpout.json", "w"), indent=2)
end = time()

print(f"time: {(end - start) / 60}mn {(end - start) % 60}s")