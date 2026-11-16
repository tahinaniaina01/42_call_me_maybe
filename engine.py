"""Function calling engine module implementing model-based function selection."""

import json
from typing import Any
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from llm_sdk import Small_LLM_Model


class FunctionCallingEngine(BaseModel):
    """Engine responsible for selecting functions and extracting their arguments."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    sdk: Small_LLM_Model
    functions: list[dict[str, Any]] = Field(default_factory=list)
    prompt_select_tpl: str = Field(default="")
    prompt_args_tpl: str = Field(default="")

    def __init__(
        self,
        sdk: Small_LLM_Model,
        prompt_select_path: str = "data/prompt_select_function.txt",
        prompt_args_path: str = "data/prompt_extract_args.txt",
        **kwargs: Any,
    ) -> None:
        """Initialize the function calling engine with model and prompt templates.

        Args:
            sdk: The small language model SDK instance.
            prompt_select_path: Path to the function selection prompt template.
            prompt_args_path: Path to the argument extraction prompt template.
            **kwargs: Additional keyword arguments for BaseModel.
        """
        with open(prompt_select_path, "r", encoding="utf-8") as f_sel:
            select_tpl = f_sel.read()
        with open(prompt_args_path, "r", encoding="utf-8") as f_args:
            args_tpl = f_args.read()

        super().__init__(
            sdk=sdk,
            prompt_select_tpl=select_tpl,
            prompt_args_tpl=args_tpl,
            **kwargs,
        )

    def select_function(self, user_prompt: str) -> str:
        """Select the best matching function name for a user request.

        Uses prefix matching to exit as soon as a single function is uniquely
        identified.

        Args:
            user_prompt: The raw user request text.

        Returns:
            The selected function name.
        """
        func_names = [str(f["name"]) for f in self.functions]
        func_list = "\n".join(
            f'- {f["name"]}: {f.get("description", "")}' for f in self.functions
        )

        prompt = (
            self.prompt_select_tpl.replace("{functions_list}", func_list).replace(
                "{user_prompt}", user_prompt
            )
        )

        ids = self.sdk.encode(prompt).tolist()[0]
        generated_ids: list[int] = []

        for _ in range(50):
            logits = self.sdk.get_logits_from_input_ids(ids)
            token_id = int(np.argmax(logits))
            ids.append(token_id)
            generated_ids.append(token_id)

            text = self.sdk.decode(generated_ids).strip().strip("\"'`")
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

        text = self.sdk.decode(generated_ids).strip().strip("\"'`")
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

    def extract_arguments(
        self, user_prompt: str, function_name: str
    ) -> dict[str, Any]:
        """Extract arguments as a dictionary from the user request.

        Args:
            user_prompt: The raw user request text.
            function_name: The target function name.

        Returns:
            Dictionary containing the extracted arguments.
        """
        func_def = next(f for f in self.functions if f["name"] == function_name)
        params: dict[str, Any] = func_def.get("parameters", {})
        if not params:
            return {}

        param_list = "\n".join(
            f'- {name} ({info.get("type", "any")})'
            for name, info in params.items()
        )
        format_items = ", ".join(
            f'"{name}": <{info.get("type", "value")}>'
            for name, info in params.items()
        )
        param_format = "{" + format_items + "}"

        prompt = (
            self.prompt_args_tpl.replace("{function_name}", function_name)
            .replace("{param_list}", param_list)
            .replace("{param_format}", param_format)
            .replace("{user_prompt}", user_prompt)
        )

        ids = self.sdk.encode(prompt).tolist()[0]
        response_ids: list[int] = []
        br = ["{"]

        for _ in range(80):
            logits = self.sdk.get_logits_from_input_ids(ids)
            token_id = int(np.argmax(logits))
            ids.append(token_id)
            response_ids.append(token_id)

            t = self.sdk.decode(token_id)
            if "{" in t:
                br.extend(["{"] * t.count("{"))
            if "}" in t:
                for _ in range(t.count("}")):
                    if br:
                        br.pop()
            if not br:
                decoded_str = '{"' + self.sdk.decode(response_ids)
                parsed_dict: dict[str, Any] = json.loads(decoded_str)
                return parsed_dict

        fallback_str = '{"' + self.sdk.decode(response_ids)
        if not fallback_str.rstrip().endswith("}"):
            fallback_str += "}"
        fallback_dict: dict[str, Any] = json.loads(fallback_str)
        return fallback_dict

    def fix_argument_types(
        self, args: dict[str, Any], function_name: str
    ) -> dict[str, Any]:
        """Validate and cast extracted arguments to their expected types.

        Only retains parameters that are explicitly defined in the function
        schema.

        Args:
            args: Raw extracted arguments dictionary.
            function_name: The target function name.

        Returns:
            Dictionary of filtered and typed arguments.
        """
        func_def = next(f for f in self.functions if f["name"] == function_name)
        params: dict[str, Any] = func_def.get("parameters", {})
        fixed: dict[str, Any] = {}

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
                elif t in ("boolean", "bool") and isinstance(
                    v, (bool, int, str)
                ):
                    if isinstance(v, str):
                        fixed[k] = v.lower() == "true"
                    else:
                        fixed[k] = bool(v)
                else:
                    fixed[k] = v
        return fixed
