"""Reference data model for citation processing."""

from typing import Annotated, List, Optional
from pydantic import AfterValidator, BaseModel, BeforeValidator, Field, model_validator

from .person import Person
from .organization import Organization
from .validators import to_str, to_list, empty_to_none, normalize, remove_empty_models


class Reference(BaseModel):
    """A reference based on the TEI biblstruct format.
    
    See: https://www.tei-c.org/release/doc/tei-p5-doc/en/html/ref-title.html
    """

    analytic_title: Optional[
        Annotated[
            str,
            BeforeValidator(to_str),
            AfterValidator(empty_to_none),
            AfterValidator(normalize),
        ]
    ] = Field(
        None,
        description="This title applies to an analytic item, such as an article, poem, or other work published as part of a larger item.",
    )
    
    monographic_title: Optional[
        Annotated[
            str,
            BeforeValidator(to_str),
            AfterValidator(empty_to_none),
            AfterValidator(normalize),
        ]
    ] = Field(
        None,
        description="This title applies to a monograph such as a book or other item considered to be a distinct publication, including single volumes of multi-volume works.",
    )
    
    journal_title: Optional[
        Annotated[
            str,
            BeforeValidator(to_str),
            AfterValidator(empty_to_none),
            AfterValidator(normalize),
        ]
    ] = Field(
        None,
        description="This title applies to any serial or periodical publication such as a journal, magazine, or newspaper.",
    )
    
    authors: Optional[
        Annotated[
            List[Person | Organization],
            BeforeValidator(to_list),
            AfterValidator(remove_empty_models),
            AfterValidator(empty_to_none),
        ]
    ] = Field(
        None,
        description="Contains the name or names of the authors, personal or corporate, of a work; for example in the same form as that provided by a recognized bibliographic name authority.",
    )
    
    editors: Optional[
        Annotated[
            List[Person | Organization],
            BeforeValidator(to_list),
            AfterValidator(remove_empty_models),
            AfterValidator(empty_to_none),
        ]
    ] = Field(
        None,
        description="Contains a secondary statement of responsibility for a bibliographic item, for example the name of an individual, institution or organization, (or of several such) acting as editor, compiler, etc.",
    )
    
    publisher: Optional[
        Annotated[
            str,
            BeforeValidator(to_str),
            AfterValidator(empty_to_none),
            AfterValidator(normalize),
        ]
    ] = Field(
        None,
        description="Contains the name of the organization responsible for the publication or distribution of a bibliographic item.",
    )
    
    translator: Optional[
        Annotated[
            Person,
            AfterValidator(empty_to_none),
        ]
    ] = Field(
        None,
        description="Contains the name of the translator of a work.",
    )
    
    publication_date: Optional[
        Annotated[
            str,
            BeforeValidator(to_str),
            AfterValidator(empty_to_none),
            AfterValidator(normalize),
        ]
    ] = Field(None, description="Contains the date of publication in any format.")
    
    publication_place: Optional[
        Annotated[
            str,
            BeforeValidator(to_str),
            AfterValidator(empty_to_none),
            AfterValidator(normalize),
        ]
    ] = Field(
        None,
        description="Contains the name of the place where a bibliographic item was published.",
    )
    
    volume: Optional[
        Annotated[
            str,
            BeforeValidator(to_str),
            AfterValidator(empty_to_none),
            AfterValidator(normalize),
        ]
    ] = Field(
        None,
        description="Defines the scope of a bibliographic reference in terms of the volume of a larger work.",
    )
    
    issue: Optional[
        Annotated[
            str,
            BeforeValidator(to_str),
            AfterValidator(empty_to_none),
            AfterValidator(normalize),
        ]
    ] = Field(
        None,
        description="Defines the scope of a bibliographic reference in terms of an issue number, or issue numbers.",
    )
    
    pages: Optional[
        Annotated[
            str,
            BeforeValidator(to_str),
            AfterValidator(empty_to_none),
            AfterValidator(normalize),
        ]
    ] = Field(
        None,
        description="Defines the scope of a bibliographic reference in terms of page numbers.",
    )
    
    cited_range: Optional[
        Annotated[
            str,
            BeforeValidator(to_str),
            AfterValidator(empty_to_none),
            AfterValidator(normalize),
        ]
    ] = Field(
        None,
        description="Defines the range of cited content, often represented by pages or other units.",
    )
    
    footnote_number: Optional[
        Annotated[
            str,
            BeforeValidator(to_str),
            AfterValidator(empty_to_none),
            AfterValidator(normalize),
        ]
    ] = Field(
        None,
        description="Contains the number of the footnote in which the reference occurs.",
    )
    
    refs: Optional[
        Annotated[
            str,
            BeforeValidator(to_str),
            AfterValidator(empty_to_none),
            AfterValidator(normalize),
        ]
    ] = Field(
        None,
        description="Defines references to another location, possibly modified by additional text or comment.",
        exclude=True,  # This means that for now it is excluded from the prompt and the evaluation (and the built-in serialization)!
    )

    @model_validator(mode="after")
    def _avoid_empty_monograph_title(self) -> "Reference":
        """TEI biblStructs require a monograph title.

        We make the life easier for the extraction model by moving the analytic title if necessary.
        """
        if (
            self.monographic_title is None
            and self.journal_title is None
            and self.analytic_title is not None
        ):
            self.monographic_title = self.analytic_title
            self.analytic_title = None

        return self 