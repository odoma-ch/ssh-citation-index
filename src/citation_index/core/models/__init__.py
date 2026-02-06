"""Core data models for citation processing."""

from .person import Person
from .organization import Organization  
from .reference import Reference
from .references import References
from .validators import (
    to_str,
    to_list,
    empty_to_none,
    normalize,
    remove_empty_models,
)
from ...utils.identifier_parser import Identifier

__all__ = [
    "Person",
    "Organization", 
    "Reference",
    "References",
    "Identifier",
    "to_str",
    "to_list", 
    "empty_to_none",
    "normalize",
    "remove_empty_models",
] 