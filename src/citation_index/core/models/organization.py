"""Organization data model for citation processing."""

from typing import Annotated, Optional
from pydantic import AfterValidator, BaseModel, BeforeValidator, Field

from .validators import to_str, empty_to_none, normalize


class Organization(BaseModel):
    """Contains information about an identifiable organization.
    
    This includes businesses, tribes, or any other grouping of people.
    """

    name: Optional[
        Annotated[
            str,
            BeforeValidator(to_str),
            AfterValidator(empty_to_none),
            AfterValidator(normalize),
        ]
    ] = Field(None, description="Contains an organizational name.") 