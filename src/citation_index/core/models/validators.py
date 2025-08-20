"""Validation functions for data models."""

from typing import Any, List, Optional
from pydantic import BaseModel


def to_str(value: Any) -> str:
    """Convert value to string and strip whitespace."""
    return str(value).strip()


def to_list(value: Any) -> List[Any]:
    """Convert value to list if it's not already a list."""
    if not isinstance(value, list):
        return [value]
    return value


def empty_to_none(value: Any) -> Optional[Any]:
    """Convert empty values to None."""
    if isinstance(value, str) and value == "":
        return None
    if isinstance(value, (tuple, list)) and (
        value == [] or value == () or value == [""] or value == ("",)
    ):
        return None
    if isinstance(value, BaseModel) and value == type(value)():
        return None
    return value


def normalize(value: Any) -> Optional[Any]:
    """Normalize strings.

    - Strip white spaces, tabs and new lines.
    - Replace tabs, new lines and multiple white spaces with one white space.
    """
    if value is None:
        return None
    if isinstance(value, tuple):
        return tuple([normalize(v) for v in value])
    if isinstance(value, list):
        return [normalize(v) for v in value]
    if isinstance(value, str):
        return " ".join(value.split())

    return value


def remove_empty_models(value: Any) -> Any:
    """Remove empty models from a list, but keep non-model objects (like strings)."""
    if not isinstance(value, list):
        return value

    new_value = []
    for v in value:
        if isinstance(v, BaseModel):
            # For BaseModel objects, only keep if they're not empty
            if v != type(v)():
                new_value.append(v)
        else:
            # For non-BaseModel objects (strings, etc.), keep them
            new_value.append(v)

    return new_value 