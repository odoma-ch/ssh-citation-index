"""Tests for data models."""

from citation_index.core.models import Person, Reference


class TestPerson:
    """Test Person model."""

    def test_person_validation(self):
        """Test Person validation and normalization."""
        person = Person(first_name="  John  ", surname="")

        assert person.first_name == "John"
        assert person.surname is None  # Empty string becomes None


class TestReference:
    """Test Reference model."""

    def test_reference_validation(self):
        """Test Reference validation logic (migrate deprecated fields to full_title)."""
        ref = Reference(analytic_title="Article Title")

        # Validator sets full_title from analytic_title when full_title is None
        assert ref.full_title == "Article Title"
        assert ref.analytic_title == "Article Title"
