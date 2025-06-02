"""Person data model for citation processing."""

from typing import Annotated, Optional
from pydantic import AfterValidator, BaseModel, BeforeValidator, Field

from .validators import to_str, empty_to_none, normalize


class Person(BaseModel):
    """Contains a proper noun or proper-noun phrase referring to a person.
    
    This includes one or more of the person's forenames, surnames, 
    honorifics, added names, etc.
    """

    first_name: Optional[
        Annotated[
            str,
            BeforeValidator(to_str),
            AfterValidator(empty_to_none),
            AfterValidator(normalize),
        ]
    ] = Field(None, description="Contains a first name, given or baptismal name.")
    
    middle_name: Optional[
        Annotated[
            str,
            BeforeValidator(to_str),
            AfterValidator(empty_to_none),
            AfterValidator(normalize),
        ]
    ] = Field(
        None, 
        description="Contains a middle name, written between a person's first and surname. It is often abbreviated."
    )
    
    surname: Optional[
        Annotated[
            str,
            BeforeValidator(to_str),
            AfterValidator(empty_to_none),
            AfterValidator(normalize),
        ]
    ] = Field(
        None,
        description="Contains a family (inherited) name of a person, as opposed to a given, baptismal, or nick name.",
    )
    
    name_link: Optional[
        Annotated[
            str,
            BeforeValidator(to_str),
            AfterValidator(empty_to_none),
            AfterValidator(normalize),
        ]
    ] = Field(
        None,
        description="Contains a connecting phrase or link used within a name but not regarded as part of it, such as 'van der' or 'of'.",
    )
    
    role_name: Optional[
        Annotated[
            str,
            BeforeValidator(to_str),
            AfterValidator(empty_to_none),
            AfterValidator(normalize),
        ]
    ] = Field(
        None,
        description="Contains a name component which indicates that the referent has a particular role or position in society, such as an official title or rank.",
    ) 