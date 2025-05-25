from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional

from pydantic import AfterValidator, BaseModel, BeforeValidator, Field, model_validator


def to_str(value: Any) -> str:
    return str(value).strip()


def to_list(value: Any) -> List[Any]:
    if not isinstance(value, list):
        return [value]
    return value


def empty_to_none(value: Any) -> Optional[Any]:
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
    """Remove empty models from a list."""
    if not isinstance(value, list):
        return value

    new_value = []
    for v in value:
        if not isinstance(v, BaseModel):
            continue
        if v == type(v)():
            continue
        new_value.append(v)

    return new_value


class Person(BaseModel):
    """Contains a proper noun or proper-noun phrase referring to a person, possibly including one or more of the person's forenames, surnames, honorifics, added names, etc."""

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
    ] = Field(None, description="Contains a middle name, written between a person's first and surname. It is often abbreviated.")
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
    

class Organization(BaseModel):
    """Contains information about an identifiable organization such as a business, a tribe, or any other grouping of people."""

    name: Optional[
        Annotated[
            str,
            BeforeValidator(to_str),
            AfterValidator(empty_to_none),
            AfterValidator(normalize),
        ]
    ] = Field(None, description="Contains an organizational name.")


class Reference(BaseModel):
    """A reference based on the TEI biblstruct format."""

    # https://www.tei-c.org/release/doc/tei-p5-doc/en/html/ref-title.html

    analytic_title: Optional[
        Annotated[
            str,
            BeforeValidator(to_str),
            AfterValidator(empty_to_none),
            AfterValidator(normalize),
        ]
    ] = Field(
        None,
        description="This title applies to an analytic item, such as an article, poem, or other work published as part "
        "of a larger item.",
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
        description="This title applies to a monograph such as a book or other item considered to be a distinct "
        "publication, including single volumes of multi-volume works.",
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
        description="This title applies to any serial or periodical publication such as a journal, magazine, or "
        "newspaper.",
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
        description="Contains the name or names of the authors, personal or corporate, of a work; for example in the "
        "same form as that provided by a recognized bibliographic name authority.",
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
        description="Contains a secondary statement of responsibility for a bibliographic item, for example the name "
        "of an individual, institution or organization, (or of several such) acting as editor, compiler, etc.",
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
        description="Contains the name of the organization responsible for the publication or distribution of a "
        "bibliographic item.",
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
        description="Defines references to another location, possibly modified by additional text or comment. ",
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


class References(list):
    def to_xml(
        self,
        file_path: Optional[str | Path] = None,
        pretty_print: bool = True,
        namespaces: Optional[Dict[str, str]] = "default",
    ) -> str:
        """Convert the references to TEI <biblStruct> elements, and optionally save them to an XML file.

        With the default namespaces the output looks like this:
        ```xml
        <TEI xmlns="http://www.tei-c.org/ns/1.0">
            <listBibl>
                <biblStruct>
                    ...
                </biblStruct>
                <biblStruct>
                    ...
            </listBibl>
            <listBibl>
                ...
        </TEI>
        ```

        Args:
            file_path: The file path to save the XML string to.
            pretty_print: Pretty print the XML?
            namespaces: The namespaces to use in the XML. By default, we use the `llamore.parser.DEFAULT_NAMESPACES`.

        Returns:
            The file path if saving to a file, or the XML string if not saving to a file.
        """
        from .parser import TeiBiblStruct

        return TeiBiblStruct(namespaces=namespaces).to_xml(
            references=self, file_path=file_path, pretty_print=pretty_print
        )

    @classmethod
    def from_xml(
        cls,
        file_path: Optional[str | Path] = None,
        xml_str: Optional[str] = None,
        namespaces: Optional[Dict[str, str]] = "default",
    ) -> "References":
        """Create References from an XML file or string that contains TEI <listBibl> with <biblStruct> elements.

        An example XML file could look like this:
        ```xml
        <TEI xmlns="http://www.tei-c.org/ns/1.0">
            ...
            <listBibl>
                <biblStruct>
                    ...
                </biblStruct>
                <biblStruct>
                    ...
            </listBibl>
            <listBibl>
                ...
        </TEI>
        ```

        Args:
            file_path: The file path to the XML file.
            xml_str: The XML string to parse.
            namespaces: The namespaces to use in the XML. By default, we use the `llamore.parser.DEFAULT_NAMESPACES`.

        Returns:
            An instance of this class, that is a list of `Reference`.
        """
        from parser import TeiBiblStruct

        list_of_list_of_references = TeiBiblStruct(namespaces=namespaces).from_xml(
            file_path=file_path, xml_str=xml_str
        )

        return cls([ref for refs in list_of_list_of_references for ref in refs])
    
    @classmethod
    def from_EXCITE_xml(cls, file_path: str) -> "References":
        """Create References from an EXCITE .txt file.

        Args:
            file_path: The file path to the EXCITE .txt file.

        Returns:
            An instance of this class, that is a list of `Reference`.
        """
        with open(file_path, "r") as file:
            lines = file.readlines()

        references = []
        for line in lines:
            references.append(Reference.from_EXCITE_xml(line))  

        return cls(references)