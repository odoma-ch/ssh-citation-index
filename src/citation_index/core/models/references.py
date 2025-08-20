"""References collection class for citation processing."""

from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING, Annotated, Any
from pydantic import BaseModel, Field, ConfigDict

from citation_index.core.models.reference import Reference


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
    def from_dict(cls, data: List[Dict[str, Any]]) -> "References":
        """Create References from a list of dictionaries."""
        references = []
        for item in data:
            if 'reference' in item:
                references.append(Reference.from_dict(item['reference']))
            else:
                references.append(Reference.from_dict(item))
        return cls(references=references) 

    @classmethod
    def from_linkedbook(cls, linkedbook_data: List[Dict[str, Any]]) -> "References":
        """Create References from LinkedBook format data.
        
        Args:
            linkedbook_data: List of LinkedBook items with 'tags', 'language', etc.
            
        Returns:
            References object with converted Reference instances.
        """
        references = []
        for item in linkedbook_data:
            # Extract tags and other metadata
            tags = item.get("tags", {})
            language = item.get("language", "")
            
            # Convert LinkedBook tags to Reference fields
            ref_data = {}
            
            # Map LinkedBook fields to Reference fields
            if "title" in tags:
                ref_data["full_title"] = tags["title"].strip(", ")
            
            if "author" in tags:
                # Split authors and clean them
                author_str = tags["author"].strip(", ")
                if author_str:
                    # Simple split by common separators, could be enhanced
                    authors = [auth.strip() for auth in author_str.split(" - ") if auth.strip()]
                    if not authors:  # If no splits found, use the whole string
                        authors = [author_str]
                    ref_data["authors"] = authors
            
            if "publicationplace" in tags:
                ref_data["publication_place"] = tags["publicationplace"].strip(", ")
            
            # Handle year - could be in 'year' or 'publicationnumber-year'
            if "year" in tags:
                ref_data["publication_date"] = tags["year"].strip(", .")
            elif "publicationnumber-year" in tags:
                ref_data["publication_date"] = tags["publicationnumber-year"].strip(", .")
            
            # Add language if available
            if language:
                ref_data["language"] = language
            
            # Create Reference from converted data
            try:
                reference = Reference.from_dict(ref_data)
                references.append(reference)
            except Exception as e:
                # Log warning but continue processing
                print(f"Warning: Could not create Reference from LinkedBook item: {e}")
                print(f"Item data: {item}")
                # Create minimal Reference as fallback
                references.append(Reference())
        
        return cls(references=references) 

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
        """Create References from an EXCITE .xml file.

        Args:
            file_path: The file path to the EXCITE .xml file.

        Returns:
            An instance of this class, that is a list of `Reference`.
        """
        # from citation_index.core.models.reference import Reference
        
        with open(file_path, "r") as file:
            lines = file.readlines()

        references = []
        for line in lines:
            # Note: This assumes Reference has a from_EXCITE_xml method
            # which would need to be implemented if this functionality is needed
            references.append(Reference.from_excite_xml(line))

        return cls(references=references)

    @classmethod
    def schema_without_excluded(cls):
        """
        Generates a JSON schema for the model, where the nested Reference objects
        do not contain fields marked with `exclude=True`.
        """
        from citation_index.core.models.reference import Reference
        
        schema = super().model_json_schema()
        
        # Pydantic may use a $ref or inline the schema for nested models.
        items_schema = schema.get("properties", {}).get("references", {}).get("items")
        
        if not items_schema:
            return schema # Should not happen, but good to be safe.
            
        if "$ref" in items_schema:
            # Case 1: The schema uses a reference (e.g. in '#/$defs/Reference').
            ref_path = items_schema["$ref"]
            ref_name = ref_path.split("/")[-1]
            if "$defs" in schema and ref_name in schema["$defs"]:
                schema["$defs"][ref_name] = Reference.schema_without_excluded()
        else:
            # Case 2: The schema is inlined.
            schema["properties"]["references"]["items"] = Reference.schema_without_excluded()
            
        # The schema we get from pydantic is 99% of the way there, but the last mile requires a bit of work.
        # First, we'll remove the `exclude` and `description` fields from the schema.
        # Then, we'll add a `name` field to the schema, which is required by the API.
        if "properties" in schema:
            for prop in ["exclude", "description"]:
                if prop in schema["properties"]:
                    del schema["properties"][prop]
        if "required" in schema:
            schema["required"] = [
                req for req in schema["required"] if req not in ["exclude"]
            ]
        # Then we add the name
        schema["name"] = cls.__name__
        return schema


# Update forward references after all models are defined
from citation_index.core.models.reference import Reference
References.model_rebuild() 

if __name__ == "__main__":
    filepath = '/Users/alex/docs/code/Odoma/citation_index/EXgoldstandard/Goldstandard_EXparser/all_xml/1181.xml'
    references = References.from_excite_xml(filepath)
    print(references)