"""Reference data model for citation processing."""

import re
from typing import Annotated, List, Optional, Dict, Tuple, Any
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
        exclude=True,
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
        exclude=True,
    )

    full_title: Optional[
        Annotated[
            str,
            BeforeValidator(to_str),
            AfterValidator(empty_to_none),
            AfterValidator(normalize),
        ]
    ] = Field(None, description="The full title of the reference. This should be the same as the monographic_title or analytic_title, or a combination of the two. ")
    
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
                List[Person | Organization | str],
                BeforeValidator(to_list),
                AfterValidator(remove_empty_models),
                AfterValidator(empty_to_none),
            ]
        ] = Field(
        default=None,
        description="Contains the name or names of the authors, personal or corporate, of a work; for example in the same form as that provided by a recognized bibliographic name authority.",
    )
    
    editors: Optional[
        Annotated[
            List[Person | Organization | str],
            BeforeValidator(to_list),
            AfterValidator(remove_empty_models),
            AfterValidator(empty_to_none),
        ]
    ] = Field(
        default=None,
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
            List[Person | str],
            BeforeValidator(to_list),
            AfterValidator(remove_empty_models),
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
    def _set_full_title(self) -> "Reference":
        if self.full_title is None:
            if self.monographic_title is not None:
                self.full_title = self.monographic_title
            elif self.analytic_title is not None:
                self.full_title = self.analytic_title
        return self

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
    def from_dict(cls, data: Dict) -> "Reference":
        """Create Reference from a dictionary."""
        return cls(**data)

    @classmethod
    def from_excite_xml(cls, xml_str: str) -> "Reference":
        """Create a Reference from an EXCITE XML string (which may be a fragment with text between tags)."""
        from lxml import etree
        import re

        # Wrap in a root element to make it parseable
        wrapped_xml = f"<root>{xml_str}</root>"

        # Use recover=True to handle non-XML text between tags
        parser = etree.XMLParser(recover=True)
        try:
            root = etree.fromstring(wrapped_xml.encode("utf-8"), parser=parser)
        except Exception as e:
            raise ValueError(f"Could not parse EXCITE XML fragment: {e}")

        # Extract authors
        authors = []
        for author in root.findall(".//author"):
            surname = author.find("surname")
            given_names = author.find("given-names")
            surname_text = surname.text.strip() if surname is not None and surname.text else None
            given_names_text = given_names.text.strip() if given_names is not None and given_names.text else None
            if surname_text or given_names_text:
                from .person import Person
                person = Person(
                    surname=surname_text,
                    first_name=given_names_text
                )
                # Only append if at least one field is not None/empty
                if person.surname or person.first_name:
                    authors.append(person)
        # Remove any None or empty authors (extra safety)
        authors = [a for a in authors if a is not None and (getattr(a, "surname", None) or getattr(a, "first_name", None))]

        # Extract editors (person or organization)
        editors = []
        for editor in root.findall(".//editor"):
            surname = editor.find("surname")
            given_names = editor.find("given-names")
            surname_text = surname.text.strip() if surname is not None and surname.text else None
            given_names_text = given_names.text.strip() if given_names is not None and given_names.text else None
            if surname_text or given_names_text:
                from .person import Person
                person = Person(
                    surname=surname_text,
                    first_name=given_names_text
                )
                if person.surname or person.first_name:
                    editors.append(person)
            elif editor.text and editor.text.strip():
                from .organization import Organization
                org = Organization(name=editor.text.strip())
                if org.name:
                    editors.append(org)
        # Remove any None or empty editors
        editors = [e for e in editors if e is not None and (getattr(e, "surname", None) or getattr(e, "first_name", None) or getattr(e, "name", None))]

        # Extract other fields
        title = root.find(".//title")
        full_title = None
        if title is not None and title.text and title.text.strip():
            full_title = title.text.strip()
        else:
            # Fallback: try to extract <title>...</title> manually from the string
            m = re.search(r"<title>(.*?)</title>", xml_str)
            if m:
                full_title = m.group(1).strip()
        if not full_title:
            full_title = "Unknown Title"
        
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
        # print(f"DEBUG: full_title = {full_title} | authors = {authors} | editors = {editors}")
        return cls(
            authors=authors if authors else None,
            editors=editors if editors else None,
            full_title=full_title,
            journal_title=source.text if source is not None else None,
            publication_date=year.text if year is not None else None,
            volume=volume.text if volume is not None else None,
            issue=issue.text if issue is not None else None,
            pages=pages,
            publisher=publisher.text if publisher is not None else None,
            publication_place=other.text if other is not None else None
        ) 

    @staticmethod
    def _extract_year(text: Optional[str]) -> Optional[str]:
        """Extract a 4-digit year when present; return None otherwise.

        Looks for 4-digit numbers in a broad historical range.
        """
        if not text:
            return None
        m = re.search(r"\b(1[5-9]\d{2}|20\d{2}|21\d{2})\b", str(text))
        if m:
            return m.group(1)
        return None

    @staticmethod
    def _canonicalize_text(text: Optional[str]) -> str:
        """Normalize free text for robust fuzzy matching."""
        if not text:
            return ""
        t = str(text).strip().upper()
        t = re.sub(r"\s+", " ", t)
        t = re.sub(r"\.+$", "", t)  # trailing periods
        return t

    @staticmethod
    def _split_gt_authors(gt_author_text: str) -> List[str]:
        """Split ground-truth author string into individual author tokens.

        Improved heuristics: split on commas, semicolons, and common conjunctions (and/e/&),
        as well as hyphen-separated lists seen in the dataset (" - ").
        Also handles various spacing and punctuation patterns.
        """
        if not gt_author_text:
            return []
        text = gt_author_text.strip()
        # Normalize unusual spaces and multiple spaces
        text = re.sub(r"\s+", " ", text)
        
        # Enhanced splitting patterns to handle more cases
        # - Standard separators: comma, semicolon
        # - Conjunctions: and, e (Italian), & 
        # - Hyphens with surrounding spaces
        # - "et al." patterns
        split_pattern = r"\s*(?:,|;|\band\b|\be\b|&|\s+-\s+|\s-\s|\bet\s+al\.?)\s*"
        parts = re.split(split_pattern, text, flags=re.IGNORECASE)
        
        # Clean up parts: remove empty strings, extra whitespace, trailing punctuation
        cleaned_parts = []
        for part in parts:
            part = part.strip()
            if part:
                # Remove trailing punctuation but preserve important punctuation like initials
                part = re.sub(r"[,;]+$", "", part)
                if part:  # Check again after cleaning
                    cleaned_parts.append(part)
        
        return cleaned_parts

    @staticmethod
    def _canonicalize_author_token(token: str) -> str:
        """Canonicalize a single author token into SURNAME INITIALS-like form.

        Improved canonicalization:
        - Uppercase and normalize spacing
        - Remove trailing punctuation but preserve initials
        - Handle common name patterns and abbreviations
        """
        if not token:
            return ""
        
        # Basic canonicalization from _canonicalize_text
        t = str(token).strip().upper()
        t = re.sub(r"\s+", " ", t)
        
        # Remove trailing periods and commas, but preserve initials like "J." within names
        t = re.sub(r"[,;]+$", "", t)  # Remove trailing commas/semicolons
        t = re.sub(r"\.+$", "", t)    # Remove trailing periods only at the end
        
        # Handle some common patterns
        # Normalize "AA. VV." (Authors Various) pattern
        t = re.sub(r"^AA\.\s*VV\.?$", "AA VV", t)
        
        # Handle bracket patterns like "[Domenico Malipiero]"
        t = re.sub(r"^\[(.*)\]$", r"\1", t)
        
        # Final cleanup
        t = t.strip()
        return t

    @classmethod
    def from_linkedbook(cls, item: Dict[str, Any]) -> Tuple["Reference", List[str]]:
        """Create a Reference from a LinkedBook JSONL item and return canonical GT authors list.

        Mapping policy:
        - title            -> full_title
        - publicationplace -> publication_place
        - publisher        -> publisher
        - volume           -> volume
        - series           -> journal_title (series treated as journal/publication series)
        - pagination       -> pages
        - year, date, or publicationnumber-year -> publication_date (best 4-digit extraction)
        - archival fields (abbreviation, archive_lib, folder, numbered_ref, ref, conjunction, 
          tomo, publicationspecifications) are ignored for structured fields
        - authors taken from tags.author and split/canonicalized for author evaluation

        Args:
            item: LinkedBook JSONL item with 'tags' containing the reference metadata

        Returns:
            Tuple of (Reference, canonical_gt_authors_list)
        """
        tags = (item.get('tags') or {}) if isinstance(item, dict) else {}

        title = tags.get('title')
        place = tags.get('publicationplace')
        publisher = tags.get('publisher')
        volume = tags.get('volume')
        series = tags.get('series')
        pagination = tags.get('pagination')
        
        # Try multiple date sources in order of preference
        year = tags.get('year')
        date = tags.get('date')
        pubnum_year = tags.get('publicationnumber-year')

        year_val = cls._extract_year(year) or cls._extract_year(date) or cls._extract_year(pubnum_year)

        ref = cls(
            full_title=title,
            publication_place=place,
            publisher=publisher,
            volume=volume,
            journal_title=series,  # Series often represents publication series/journal
            pages=pagination,
            publication_date=year_val,
        )

        gt_author_text = tags.get('author', '')
        gt_authors = [a for a in cls._split_gt_authors(gt_author_text)]

        return ref, gt_authors

    @classmethod
    def schema_without_excluded(cls):
        """Generates a JSON schema for the model, excluding fields marked with `exclude=True`."""
        schema = cls.model_json_schema()
        excluded_fields = {
            field_name
            for field_name, field_info in cls.model_fields.items()
            if field_info.exclude
        }

        if 'properties' in schema:
            for field in excluded_fields:
                schema['properties'].pop(field, None)
        
        if 'required' in schema:
            schema['required'] = [
                req for req in schema['required'] if req not in excluded_fields
            ]
            if not schema['required']:
                schema.pop('required')
        
        schema['name'] = cls.__name__
                
        return schema 