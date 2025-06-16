"""References collection class for citation processing."""

from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING, Annotated
from pydantic import BaseModel, Field, ConfigDict

if TYPE_CHECKING:
    from .reference import Reference


class References(BaseModel):
    """A collection of Reference objects with serialization capabilities."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    references: List["Reference"] = Field(
        default_factory=list,
        description="List of Reference objects"
    )

    def __iter__(self):
        return iter(self.references)

    def __len__(self):
        return len(self.references)

    def __getitem__(self, index):
        return self.references[index]

    def append(self, item: "Reference"):
        self.references.append(item)

    def extend(self, items: List["Reference"]):
        self.references.extend(items)

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
            namespaces: The namespaces to use in the XML. By default, we use the DEFAULT_NAMESPACES.

        Returns:
            The file path if saving to a file, or the XML string if not saving to a file.
        """
        from ..parsers.tei_bibl_parser import TeiBiblParser

        return TeiBiblParser(namespaces=namespaces).to_xml(
            references=self.references, file_path=file_path, pretty_print=pretty_print
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
            namespaces: The namespaces to use in the XML. By default, we use the DEFAULT_NAMESPACES.

        Returns:
            An instance of this class, that is a list of `Reference`.
        """
        from ..parsers.tei_bibl_parser import TeiBiblParser

        list_of_list_of_references = TeiBiblParser(namespaces=namespaces).from_xml(
            file_path=file_path, xml_str=xml_str
        )

        return cls(references=[ref for refs in list_of_list_of_references for ref in refs])

    @classmethod
    def from_excite_xml(cls, file_path: str) -> "References":
        """Create References from an EXCITE .txt file.

        Args:
            file_path: The file path to the EXCITE .txt file.

        Returns:
            An instance of this class, that is a list of `Reference`.
        """
        from .reference import Reference
        
        with open(file_path, "r") as file:
            lines = file.readlines()

        references = []
        for line in lines:
            # Note: This assumes Reference has a from_EXCITE_xml method
            # which would need to be implemented if this functionality is needed
            references.append(Reference.from_excite_xml(line))

        return cls(references=references)


# Update forward references after all models are defined
from .reference import Reference
References.model_rebuild() 

if __name__ == "__main__":
    filepath = '../../EXgoldstandard/Goldstandard_EXparser/all_xml/1181.xml'
    references = References.from_excite_xml(filepath)
    print(references)