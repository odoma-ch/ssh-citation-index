import logging
from pathlib import Path
from typing import Dict, List, Literal, Optional

from lxml import etree

from reference import Organization, Person, Reference

_LOGGER = logging.getLogger(__name__)

DEFAULT_NAMESPACES = {None: "http://www.tei-c.org/ns/1.0"}


class TeiBiblStruct:
    """Read and write TEI BiblStruct formatted references.

    Args:
        namespaces: XML namespaces. By default, we use the `llamore.parser.DEFAULT_NAMESPACES`.
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
        """Extract a bibliogrpahic scope with a given 'unit' attribute from an Element"""
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
        elif authedit.text:
            surname = authedit.text.strip()
        else:
            first_name = self._find_all_and_join_text(
                authedit, "forename[@type='first']"
            )
            middle_name = self._find_all_and_join_text(
                authedit, "forename[@type='middle']"
            )
            if first_name is None and middle_name is None:
                first_name = self._find_all_and_join_text(authedit, "forename")
            surname = self._find_all_and_join_text(authedit, "surname")
            name_link = self._find_all_and_join_text(authedit, "nameLink")
            role_name = self._find_all_and_join_text(authedit, "roleName")

        person = Person(
            first_name=first_name,
            middle_name=middle_name,
            surname=surname,
            name_link=name_link,
            role_name=role_name,
        )

        if person == Person():
            return None

        return person

    def _find_organization(self, authedit: etree._Element) -> Optional[Organization]:
        organization = self._find_all_and_join_text(authedit, "orgName")

        if organization:
            return Organization(name=organization)

        return None

    def _find_all_and_join_text(
        self, element: etree._Element, tag: str, separator: str = " "
    ) -> Optional[str]:
        elements = element.findall(tag, namespaces=self._namespaces)
        texts = []
        for el in elements:
            if el.text:
                texts.append(el.text)

        if texts:
            return separator.join(texts).strip()
        return None

    def from_xml(
        self,
        file_path: Optional[str | Path] = None,
        xml_str: Optional[str] = None,
        n: Optional[int] = None,
    ) -> List[List[Reference]]:
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
            n: The maximum number of bibliographic lists (<listBibl>) to process.

        Returns:
            A list of lists of `Reference`.
        """
        if file_path is None and xml_str is None:
            raise ValueError("Please pass in either a 'file_path' or a 'xml_str'!")

        if file_path is not None and xml_str is not None:
            raise ValueError(
                "Please pass in either a 'file_path' or a 'xml_str', not both."
            )

        if file_path is not None:
            tree = etree.parse(str(file_path))
        else:
            tree = etree.fromstring(xml_str)

        list_bibls = tree.findall(".//listBibl", namespaces=self._namespaces)

        list_of_references = []
        for list_bibl in list_bibls[:n]:
            references = self.to_references(list_bibl, raise_empty_error=False)
            list_of_references.append(references)

        return list_of_references

    def to_xml(
        self,
        references: Reference | List[Reference] | List[List[Reference]],
        file_path: Optional[str | Path] = None,
        pretty_print: bool = True,
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
            references: A reference, a list of references, or a list of lists of references.
            file_path: The file path to save the XML string to.
            pretty_print: Pretty print the XML?

        Returns:
            The file path if saving to a file, or the XML string if not saving to a file.
        """
        if isinstance(references, list):
            if len(references) == 0 or isinstance(references[0], Reference):
                references = [references]
        elif isinstance(references, Reference):
            references = [[references]]

        root = etree.Element("TEI", nsmap=self._namespaces)
        for reference_list in references:
            list_bibl = etree.SubElement(root, "listBibl")
            for reference in reference_list:
                bibl_struct = etree.SubElement(
                    list_bibl,
                    "biblStruct",
                    attrib={"source": f"fn{reference.footnote_number}"}
                    if reference.footnote_number
                    else None,
                    nsmap=self._namespaces,
                )
                analytic, monogr = None, None
                if reference.analytic_title:
                    analytic = etree.SubElement(bibl_struct, "analytic")
                    title = etree.SubElement(analytic, "title", attrib={"level": "a"})
                    title.text = reference.analytic_title
                    if reference.authors:
                        self._add_persons_and_organizations(
                            analytic, reference.authors, "author"
                        )

                if reference.monographic_title:
                    monogr = self._get_or_add_subelement(bibl_struct, "monogr")
                    title = etree.SubElement(monogr, "title", attrib={"level": "m"})
                    title.text = reference.monographic_title

                if reference.journal_title:
                    monogr = self._get_or_add_subelement(bibl_struct, "monogr")
                    title = etree.SubElement(monogr, "title", attrib={"level": "j"})
                    title.text = reference.journal_title

                if analytic is None and reference.authors:
                    monogr = self._get_or_add_subelement(bibl_struct, "monogr")
                    self._add_persons_and_organizations(
                        monogr, reference.authors, "author"
                    )

                if reference.editors:
                    monogr = self._get_or_add_subelement(bibl_struct, "monogr")
                    self._add_persons_and_organizations(
                        monogr, reference.editors, "editor"
                    )

                if reference.translator:
                    monogr = self._get_or_add_subelement(bibl_struct, "monogr")
                    self._add_persons_and_organizations(
                        monogr, [reference.translator], "editor", attrib={"role": "translator"}
                    )

                if reference.publisher:
                    monogr = self._get_or_add_subelement(bibl_struct, "monogr")
                    imprint = self._get_or_add_subelement(monogr, "imprint")
                    publisher = etree.SubElement(imprint, "publisher")
                    publisher.text = reference.publisher

                # <imprint><date/><imprint> is mandatory
                monogr = self._get_or_add_subelement(bibl_struct, "monogr")
                imprint = self._get_or_add_subelement(monogr, "imprint")
                date = etree.SubElement(imprint, "date")
                date.text = reference.publication_date or ""

                if reference.publication_place:
                    monogr = self._get_or_add_subelement(bibl_struct, "monogr")
                    imprint = self._get_or_add_subelement(monogr, "imprint")
                    for place in reference.publication_place.split(","):
                        if place:
                            pubPlace = etree.SubElement(imprint, "pubPlace")
                            pubPlace.text = place.strip()

                if reference.volume:
                    monogr = self._get_or_add_subelement(bibl_struct, "monogr")
                    biblScope = etree.SubElement(
                        monogr, "biblScope", attrib={"unit": "volume"}
                    )
                    biblScope.text = reference.volume

                if reference.issue:
                    monogr = self._get_or_add_subelement(bibl_struct, "monogr")
                    biblScope = etree.SubElement(
                        monogr, "biblScope", attrib={"unit": "issue"}
                    )
                    biblScope.text = reference.issue

                if reference.pages:
                    monogr = self._get_or_add_subelement(bibl_struct, "monogr")
                    biblScope = etree.SubElement(
                        monogr, "biblScope", attrib={"unit": "page"}
                    )
                    biblScope.text = reference.pages

                if reference.cited_range:
                    cr = etree.SubElement(bibl_struct, "citedRange")
                    cr.text = reference.cited_range

                if reference.refs:
                    ref = etree.SubElement(bibl_struct, "ref")
                    ref.text = reference.refs

        if file_path is not None:
            et = etree.ElementTree(root)
            et.write(str(file_path), pretty_print=pretty_print)

            return str(file_path)
        else:
            return etree.tostring(root, encoding="unicode", pretty_print=pretty_print)

    def _add_persons_and_organizations(
        self,
        element: etree._Element,
        persons_and_organizations: List[Person | Organization],
        author_or_editor: Literal["author", "editor"] = "author",
        attrib: Optional[Dict[str, str]] = None,
    ):
        for pers_or_orga in persons_and_organizations:
            authitor = etree.SubElement(element, author_or_editor, attrib=attrib)

            if isinstance(pers_or_orga, Organization):
                self._add_organization(pers_or_orga, authitor)
            else:
                self._add_person(pers_or_orga, authitor)

    def _add_person(self, person: Person, authitor: etree._Element):
        persName = etree.SubElement(authitor, "persName")
        if person.first_name is not None:
            first_name = etree.SubElement(
                persName, "forename", attrib={"type": "first"}
            )
            first_name.text = person.first_name
        if person.middle_name is not None:
            middle_name = etree.SubElement(
                persName, "forename", attrib={"type": "middle"}
            )
            middle_name.text = person.middle_name
        if person.surname is not None:
            surname = etree.SubElement(persName, "surname")
            surname.text = person.surname
        if person.name_link is not None:
            name_link = etree.SubElement(persName, "nameLink")
            name_link.text = person.name_link
        if person.role_name is not None:
            role_name = etree.SubElement(persName, "roleName")
            role_name.text = person.role_name

    def _add_organization(self, organization: Organization, authitor: etree._Element):
        orgName = etree.SubElement(authitor, "orgName")
        orgName.text = organization.name

    def _get_or_add_subelement(
        self, element: etree._Element, tag: str
    ) -> etree._Element:
        """Get the subelement or add it if it does not exist."""
        subelement = element.find(tag)
        if subelement is not None:
            return subelement

        return etree.SubElement(element, tag)
