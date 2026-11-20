"""Argument parsing and strict JSON validation."""

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator


DEFAULT_FUNCTIONS_DEFINITION = "data/input/functions_definition.json"
DEFAULT_INPUT = "data/input/function_calling_tests.json"
DEFAULT_OUTPUT = "data/output/outpout.json"


class CliArguments(BaseModel):
    """Validated command-line arguments."""

    model_config = ConfigDict(extra="forbid")

    functions_definition: str = DEFAULT_FUNCTIONS_DEFINITION
    input: str = DEFAULT_INPUT
    output: str = DEFAULT_OUTPUT

    @field_validator("functions_definition", "input", "output")
    @classmethod
    def validate_non_empty_path(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("path must not be empty")
        return value


class TypeDefinition(BaseModel):
    """Parameter or return type definition."""

    model_config = ConfigDict(extra="forbid")

    type: str

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("type must be a non-empty string")
        return value


class FunctionDefinition(BaseModel):
    """Definition of one callable function."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, TypeDefinition]
    returns: TypeDefinition

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("name must be a non-empty string")
        return value

    @field_validator("parameters")
    @classmethod
    def validate_parameters(
        cls, value: dict[str, TypeDefinition]
    ) -> dict[str, TypeDefinition]:
        invalid_names = [name for name in value if not name.strip()]
        if invalid_names:
            raise ValueError("parameters contain an invalid parameter name")
        return value


class PromptItem(BaseModel):
    """One user prompt entry."""

    model_config = ConfigDict(extra="forbid")

    prompt: str

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must be a non-empty string")
        return value


class Parser:
    """Parse CLI arguments and validate project JSON files."""

    @staticmethod
    def parse(argv: Sequence[str] | None = None) -> CliArguments:
        parser = argparse.ArgumentParser(
            description="Run function-calling tests from JSON input files."
        )
        parser.add_argument(
            "--functions_definition",
            "--functions-definition",
            default=DEFAULT_FUNCTIONS_DEFINITION,
            help=(
                "Function definitions JSON file "
                f"(default: {DEFAULT_FUNCTIONS_DEFINITION})"
            ),
        )
        parser.add_argument(
            "--input",
            default=DEFAULT_INPUT,
            help=f"User prompts JSON file (default: {DEFAULT_INPUT})",
        )
        parser.add_argument(
            "--output",
            default=DEFAULT_OUTPUT,
            help=f"Output JSON file (default: {DEFAULT_OUTPUT})",
        )
        namespace = parser.parse_args(argv)
        try:
            return CliArguments.model_validate(vars(namespace))
        except ValidationError as error:
            parser.error(str(error))
            raise AssertionError("argparse.error unexpectedly returned")

    @staticmethod
    def load_json(path: str, description: str) -> Any:
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"{description} file not found: {file_path}")

        try:
            with file_path.open("r", encoding="utf-8") as file:
                return json.load(file)
        except json.JSONDecodeError as error:
            raise json.JSONDecodeError(
                f"Invalid JSON in {description} file '{file_path}': {error.msg}",
                error.doc,
                error.pos,
            ) from error
        except OSError as error:
            raise OSError(
                f"Unable to read {description} file '{file_path}': {error}"
            ) from error

    @staticmethod
    def validate_exact_keys(
        value: dict[str, Any],
        expected_keys: set[str],
        location: str,
    ) -> None:
        actual_keys = set(value)
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        if missing or extra:
            details: list[str] = []
            if missing:
                details.append(f"missing keys: {', '.join(missing)}")
            if extra:
                details.append(f"unexpected keys: {', '.join(extra)}")
            raise ValueError(f"{location}: {'; '.join(details)}")

    @classmethod
    def validate_functions(cls, data: Any) -> list[dict[str, Any]]:
        if not isinstance(data, list):
            raise ValueError("functions_definition must contain a JSON list")
        if not data:
            raise ValueError(
                "functions_definition must contain at least one function"
            )

        try:
            functions = [
                FunctionDefinition.model_validate(item)
                for item in data
            ]
        except ValidationError as error:
            raise ValueError(
                f"Invalid functions_definition: {error}"
            ) from error

        names = [function.name for function in functions]
        duplicate_names = sorted({name for name in names if names.count(name) > 1})
        if duplicate_names:
            raise ValueError(
                "functions_definition contains duplicate function names: "
                + ", ".join(duplicate_names)
            )

        return [
            function.model_dump(exclude_none=False)
            for function in functions
        ]

    @classmethod
    def validate_prompts(cls, data: Any) -> list[dict[str, str]]:
        if not isinstance(data, list):
            raise ValueError("input must contain a JSON list")

        try:
            prompts = [PromptItem.model_validate(item) for item in data]
        except ValidationError as error:
            raise ValueError(f"Invalid input: {error}") from error
        return [prompt.model_dump() for prompt in prompts]

    @classmethod
    def validate_extracted_arguments(
        cls,
        args: dict[str, Any],
        function_name: str,
        function_definitions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        function = next(
            (item for item in function_definitions if item["name"] == function_name),
            None,
        )
        if function is None:
            raise ValueError(f"Selected unknown function: {function_name}")

        cls.validate_exact_keys(
            args,
            set(function["parameters"]),
            f"arguments for {function_name}",
        )
        return args

    @staticmethod
    def write_results(path: str, results: list[dict[str, Any]]) -> None:
        output_path = Path(path)
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as file:
                json.dump(results, file, indent=2, ensure_ascii=False)
                file.write("\n")
        except OSError as error:
            raise OSError(
                f"Unable to write output file '{output_path}': {error}"
            ) from error
