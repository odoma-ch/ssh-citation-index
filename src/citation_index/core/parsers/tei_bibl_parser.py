"""TEI BiblStruct parser for citation processing."""

import logging
from pathlib import Path
from typing import Dict, List, Literal, Optional

from lxml import etree

from ..models import Reference, Person, Organization

_LOGGER = logging.getLogger(__name__)

DEFAULT_NAMESPACES = {None: "http://www.tei-c.org/ns/1.0"}


class TeiBiblParser:
    """Read and write TEI BiblStruct formatted references.

    Args:
        namespaces: XML namespaces. By default, we use the DEFAULT_NAMESPACES.
    """

    def __init__(self, namespaces: Optional[Dict[str, str]] = "default"):
        self._namespaces = namespaces
        if namespaces == "default":
            self._namespaces = DEFAULT_NAMESPACES

    def to_references(
        self, bibl_struct_or_list: etree._Element, raise_empty_error: bool = True
    ) -> List[Reference]:
        """Turn a TEI <listBibl> or <biblStruct> XML tag into `Reference`s.

        Args:
            bibl_struct_or_list: The TEI <listBibl> or <biblStruct> XML element.
            raise_empty_error: Raise an error if there are empty references?

        Returns:
            A list of `Reference`s.
        """
        tag = etree.QName(bibl_struct_or_list).localname
        if tag == "listBibl":
            bibl_structs = bibl_struct_or_list.findall(
                "biblStruct", namespaces=self._namespaces
            )
            references = [
                self._to_reference(bibl_struct, raise_empty_error=raise_empty_error)
                for bibl_struct in bibl_structs
            ]
        elif tag == "biblStruct":
            references = [
                self._to_reference(
                    bibl_struct_or_list, raise_empty_error=raise_empty_error
                )
            ]
        else:
            raise ValueError(
                f"Can only process elements with tags 'listBibl' or 'biblStruct', but got '{tag}'"
            )

        return [ref for ref in references if ref is not None]

    def _to_reference(
        self, bibl_struct: etree._Element, raise_empty_error: bool = True
    ) -> Optional[Reference]:
        """Turn a TEI <biblStruct> XML element into a Reference instance.

        Args:
            bibl_struct: The TEI <biblStruct> XML element.
            raise_empty_error: Raise an error if it's an empty reference?

        Returns:
            A `Reference` instance or `None` if it's an empty reference.
        """
        analytic_title = self._find_all_and_join_text(
            bibl_struct, ".//title[@level='a']"
        )
        monographic_title = self._find_all_and_join_text(
            bibl_struct, ".//title[@level='m']"
        )
        journal_title = self._find_all_and_join_text(
            bibl_struct, ".//title[@level='j']"
        )
        authors = self._find_persons_and_organizations(bibl_struct, "author")
        editors = self._find_persons_and_organizations(bibl_struct, "editor")
        translator = self._find_translator(bibl_struct)
        publisher = self._find_all_and_join_text(bibl_struct, ".//publisher")
        publication_date = self._find_all_and_join_text(bibl_struct, ".//date")
        pages = self._find_scope(bibl_struct, "page")
        volume = self._find_scope(bibl_struct, "volume")
        issue = self._find_scope(bibl_struct, "issue")

        cited_range = self._find_all_and_join_text(bibl_struct, ".//citedRange")

        publication_place = self._find_all_and_join_text(bibl_struct, ".//pubPlace", separator=", ")

        footnote_number = bibl_struct.attrib.get("source", "")[2:]

        refs = self._find_and_join_all_refs(bibl_struct)

        reference = Reference(
            analytic_title=analytic_title,
            authors=authors,
            monographic_title=monographic_title,
            journal_title=journal_title,
            editors=editors,
            publisher=publisher,
            translator=translator,
            publication_date=publication_date,
            publication_place=publication_place,
            volume=volume,
            issue=issue,
            pages=pages,
            cited_range=cited_range,
            footnote_number=footnote_number,
            refs=refs,
        )
        if reference == Reference():
            _LOGGER.debug("Empty Reference")
            reference = None

        if reference is None and raise_empty_error:
            raise ValueError("Empty Reference")

        return reference

    def _find_and_join_all_refs(self, element: etree._Element) -> Optional[str]:
        refs = element.findall(".//ref", namespaces=self._namespaces)
        joined_refs = " ".join(
            ["".join(ref.itertext()).strip() for ref in refs]
        ).strip()

        return joined_refs or None

    def _find_scope(
        self, element: etree._Element, unit: str = "volume"
    ) -> Optional[str]:
        """Extract a bibliographic scope with a given 'unit' attribute from an Element"""
        scope = getattr(
            element.find(f".//biblScope[@unit='{unit}']", namespaces=self._namespaces),
            "text",
            None,
        )
        return scope

    def _find_persons_and_organizations(
        self,
        element: etree._Element,
        author_or_editor: Literal["author", "editor"] = "author",
    ) -> List[Person | Organization]:
        """Extract all persons/organizations from an Element.

        Args:
            element: The TEI XML element.
            author_or_editor: Do the persons or organizations belong to the <author> or <editor> element?

        Returns:
            A list with all persons or organizations.
        """
        persons_and_organizations = []
        authors_or_editors = element.findall(
            f".//{author_or_editor}", namespaces=self._namespaces
        )
        for authedit in authors_or_editors:
            # translators have their own field
            if authedit.attrib.get("role") == "translator":
                continue
            if person := self._find_person(authedit):
                persons_and_organizations.append(person)
            if organization := self._find_organization(authedit):
                persons_and_organizations.append(organization)

        return persons_and_organizations

    def _find_translator(
        self, element: etree._Element
    ) -> Optional[Person]:
        """Extract the translator from an Element."""
        translator = element.find(".//editor[@role='translator']", namespaces=self._namespaces)
        if translator is not None:
            return self._find_person(translator)
        return None

    def _find_person(self, authedit: etree._Element) -> Optional[Person]:
        first_name, middle_name, surname, name_link, role_name = (
            None,
            None,
            None,
            None,
            None,
        )

        person = authedit.find("persName", namespaces=self._namespaces)
        if person is not None:
            first_name = self._find_all_and_join_text(person, "forename[@type='first']")
            middle_name = self._find_all_and_join_text(
                person, "forename[@type='middle']"
            )
            if first_name is None and middle_name is None:
                first_name = self._find_all_and_join_text(person, "forename")
            surname = self._find_all_and_join_text(person, "surname")
            name_link = self._find_all_and_join_text(person, "nameLink")
            role_name = self._find_all_and_join_text(person, "roleName")

            if first_name or middle_name or surname or name_link or role_name:
                return Person(
                    first_name=first_name,
                    middle_name=middle_name,
                    surname=surname,
                    name_link=name_link,
                    role_name=role_name,
                )

        return None

    def _find_organization(self, authedit: etree._Element) -> Optional[Organization]:
        org_name = self._find_all_and_join_text(authedit, "orgName")
        if org_name:
            return Organization(name=org_name)
        return None

    def _find_all_and_join_text(
        self, element: etree._Element, tag: str, separator: str = " "
    ) -> Optional[str]:
        """Find all elements and join their text content"""
        elements = element.findall(tag, namespaces=self._namespaces)
        texts = [elem.text for elem in elements if elem.text]
        if texts:
            return separator.join(texts).strip()
        return None

    def from_xml(
        self,
        file_path: Optional[str | Path] = None,
        xml_str: Optional[str] = None,
        n: Optional[int] = None,
    ) -> List[List[Reference]]:
        """Parse XML file or string into References.

        Args:
            file_path: Path to XML file
            xml_str: XML string to parse
            n: Maximum number of references to parse

        Returns:
            List of lists of References
        """
        if file_path is not None:
            tree = etree.parse(str(file_path))
        elif xml_str is not None:
            tree = etree.fromstring(xml_str)
        else:
            raise ValueError("Either file_path or xml_str must be provided")

        root = tree.getroot() if hasattr(tree, 'getroot') else tree
        list_bibls = root.findall(".//listBibl", namespaces=self._namespaces)

        references_lists = []
        processed_count = 0
        
        for list_bibl in list_bibls:
            if n is not None and processed_count >= n:
                break
            
            references = self.to_references(list_bibl)
            references_lists.append(references)
            processed_count += len(references)

        return references_lists

    def to_xml(
        self,
        references,  # Can be Reference, List[Reference], or List[List[Reference]]
        file_path: Optional[str | Path] = None,
        pretty_print: bool = True,
    ) -> str:
        """Convert references to TEI XML format.

        Args:
            references: Reference | List[Reference] | List[List[Reference]]
            file_path: Optional file path to save XML
            pretty_print: Whether to format XML nicely

        Returns:
            XML string (also written to file if file_path provided)
        """
        from lxml import etree
        from ..models import Reference

        # Normalize to list of lists
        if isinstance(references, Reference):
            refs_ll = [[references]]
        elif isinstance(references, list):
            if len(references) > 0 and isinstance(references[0], Reference):
                refs_ll = [references]
            else:
                refs_ll = references  # assume already List[List[Reference]]
        else:
            refs_ll = []

        NSMAP = {None: self._namespaces.get(None, "http://www.tei-c.org/ns/1.0")}
        tei = etree.Element("TEI", nsmap=NSMAP)

        for refs in refs_ll:
            list_bibl = etree.SubElement(tei, "listBibl")
            for ref in refs:
                bibl_struct = etree.SubElement(list_bibl, "biblStruct")

                # Analytic (article in a journal or part of monograph)
                if ref.analytic_title:
                    analytic = etree.SubElement(bibl_struct, "analytic")
                    title_a = etree.SubElement(analytic, "title")
                    title_a.set("level", "a")
                    title_a.text = ref.analytic_title
                    if ref.authors:
                        for author in ref.authors:
                            _append_author_or_org(analytic, author)

                # Monographic (book/journal issue)
                monogr = etree.SubElement(bibl_struct, "monogr")
                if ref.monographic_title:
                    title_m = etree.SubElement(monogr, "title")
                    title_m.set("level", "m")
                    title_m.text = ref.monographic_title
                if ref.journal_title:
                    title_j = etree.SubElement(monogr, "title")
                    title_j.set("level", "j")
                    title_j.text = ref.journal_title

                if ref.editors:
                    for editor in ref.editors:
                        _append_author_or_org(monogr, editor, tag="editor")

                if ref.publisher:
                    publisher = etree.SubElement(monogr, "publisher")
                    publisher.text = ref.publisher
                if ref.publication_place:
                    pub_place = etree.SubElement(monogr, "pubPlace")
                    pub_place.text = ref.publication_place
                if ref.translator:
                    _append_author_or_org(monogr, ref.translator, tag="editor", role="translator")

                # Imprint / issuance details
                if any([ref.publication_date, ref.volume, ref.issue, ref.pages]):
                    imprint = etree.SubElement(monogr, "imprint")
                    if ref.publication_date:
                        date = etree.SubElement(imprint, "date")
                        date.text = ref.publication_date
                    if ref.volume:
                        vol = etree.SubElement(imprint, "biblScope")
                        vol.set("unit", "volume")
                        vol.text = ref.volume
                    if ref.issue:
                        iss = etree.SubElement(imprint, "biblScope")
                        iss.set("unit", "issue")
                        iss.text = ref.issue
                    if ref.pages:
                        pgs = etree.SubElement(imprint, "biblScope")
                        pgs.set("unit", "page")
                        pgs.text = ref.pages

                if ref.cited_range:
                    cr = etree.SubElement(bibl_struct, "citedRange")
                    cr.text = ref.cited_range

        xml_bytes = etree.tostring(tei, pretty_print=pretty_print, encoding="utf-8", xml_declaration=True)
        xml_str = xml_bytes.decode("utf-8")
        if file_path is not None:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            Path(file_path).write_text(xml_str, encoding="utf-8")
        return xml_str


def _append_author_or_org(parent, entity, tag: str = "author", role: str | None = None):
    """Append a person or organization element under parent."""
    from ..models import Person, Organization
    from lxml import etree

    el = etree.SubElement(parent, tag)
    if role:
        el.set("role", role)

    if isinstance(entity, Person):
        pers = etree.SubElement(el, "persName")
        if entity.first_name:
            fn = etree.SubElement(pers, "forename")
            fn.set("type", "first")
            fn.text = entity.first_name
        if entity.middle_name:
            mn = etree.SubElement(pers, "forename")
            mn.set("type", "middle")
            mn.text = entity.middle_name
        if entity.surname:
            sn = etree.SubElement(pers, "surname")
            sn.text = entity.surname
        if entity.name_link:
            nl = etree.SubElement(pers, "nameLink")
            nl.text = entity.name_link
        if entity.role_name:
            rn = etree.SubElement(pers, "roleName")
            rn.text = entity.role_name
    elif isinstance(entity, Organization):
        org = etree.SubElement(el, "orgName")
        org.text = entity.name
