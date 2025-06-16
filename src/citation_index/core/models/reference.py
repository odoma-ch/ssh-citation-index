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

    @classmethod
    def from_excite_xml(cls, xml_str: str) -> "Reference":
        """Create a Reference from an EXCITE XML string.

        The EXCITE XML format looks like this:
        ```xml
        <author><surname>Adams</surname>, <given-names>J. S.</given-names></author> (<year>1965</year>). <title>Inequity in social exchange</title>. In <editor>L. Berkowitz</editor> (Ed.), <source>Advances in experimental social psychology</source> (pp. <fpage>267</fpage> - <lpage>99</lpage>). <other>New York</other>: <publisher>Academic Press</publisher>.
        ```

        Args:
            xml_str: The EXCITE XML string to parse.

        Returns:
            A Reference instance.
        """
        from lxml import etree

        # Wrap the XML fragment in a root element to make it valid XML
        wrapped_xml = f"<root>{xml_str}</root>"

        # Parse the XML string
        try:
            root = etree.fromstring(wrapped_xml)
        except etree.XMLSyntaxError:
            # If the XML is not well-formed, try to fix common issues
            wrapped_xml = wrapped_xml.replace("&", "&amp;")  # Fix unescaped ampersands
            try:
                root = etree.fromstring(wrapped_xml)
            except etree.XMLSyntaxError as e:
                # If still not well-formed, try to clean up the XML
                import re
                # Remove any non-XML content between tags
                cleaned_xml = re.sub(r'>\s*([^<]+?)\s*<', '><', wrapped_xml)
                # Remove any remaining non-XML content
                cleaned_xml = re.sub(r'[^<]*<', '<', cleaned_xml)
                cleaned_xml = re.sub(r'>[^>]*', '>', cleaned_xml)
                root = etree.fromstring(cleaned_xml)

        # Extract authors
        authors = []
        for author in root.findall(".//author"):
            surname = author.find("surname")
            given_names = author.find("given-names")
            if surname is not None or given_names is not None:
                from .person import Person
                authors.append(Person(
                    surname=surname.text if surname is not None else None,
                    first_name=given_names.text if given_names is not None else None
                ))

        # Extract editors
        editors = []
        for editor in root.findall(".//editor"):
            surname = editor.find("surname")
            given_names = editor.find("given-names")
            if surname is not None or given_names is not None:
                from .person import Person
                editors.append(Person(
                    surname=surname.text if surname is not None else None,
                    first_name=given_names.text if given_names is not None else None
                ))

        # Extract other fields
        title = root.find(".//title")
        source = root.find(".//source")
        year = root.find(".//year")
        volume = root.find(".//volume")
        issue = root.find(".//issue")
        fpage = root.find(".//fpage")
        lpage = root.find(".//lpage")
        publisher = root.find(".//publisher")
        other = root.find(".//other")

        # Create pages string if both fpage and lpage exist
        pages = None
        if fpage is not None and lpage is not None:
            pages = f"{fpage.text}-{lpage.text}"
        elif fpage is not None:
            pages = fpage.text

        return cls(
            authors=authors if authors else None,
            editors=editors if editors else None,
            analytic_title=title.text if title is not None else None,
            journal_title=source.text if source is not None else None,
            publication_date=year.text if year is not None else None,
            volume=volume.text if volume is not None else None,
            issue=issue.text if issue is not None else None,
            pages=pages,
            publisher=publisher.text if publisher is not None else None,
            publication_place=other.text if other is not None else None
        ) 