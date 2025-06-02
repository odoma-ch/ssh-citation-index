"""Tests for data models."""

import pytest
from citation_index.core.models import Person, Organization, Reference, References


class TestPerson:
    """Test Person model."""
    
    def test_create_person(self):
        """Test creating a Person instance."""
        person = Person(
            first_name="John",
            middle_name="Q",
            surname="Doe",
            name_link="van",
            role_name="Dr."
        )
        
        assert person.first_name == "John"
        assert person.middle_name == "Q"
        assert person.surname == "Doe"
        assert person.name_link == "van"
        assert person.role_name == "Dr."
    
    def test_person_validation(self):
        """Test Person validation and normalization."""
        person = Person(first_name="  John  ", surname="")
        
        assert person.first_name == "John"
        assert person.surname is None  # Empty string becomes None
    
    def test_empty_person(self):
        """Test creating an empty Person."""
        person = Person()
        
        assert person.first_name is None
        assert person.surname is None


class TestOrganization:
    """Test Organization model."""
    
    def test_create_organization(self):
        """Test creating an Organization instance."""
        org = Organization(name="MIT Press")
        
        assert org.name == "MIT Press"
    
    def test_empty_organization(self):
        """Test creating an empty Organization."""
        org = Organization()
        
        assert org.name is None


class TestReference:
    """Test Reference model."""
    
    def test_create_reference(self):
        """Test creating a Reference instance."""
        person = Person(first_name="John", surname="Doe")
        org = Organization(name="MIT Press")
        
        ref = Reference(
            analytic_title="Sample Article",
            monographic_title="Sample Book", 
            authors=[person, org],
            publication_date="2023"
        )
        
        assert ref.analytic_title == "Sample Article"
        assert ref.monographic_title == "Sample Book"
        assert len(ref.authors) == 2
        assert ref.publication_date == "2023"
    
    def test_reference_validation(self):
        """Test Reference validation logic."""
        # Test the monograph title validation
        ref = Reference(analytic_title="Article Title")
        
        # Should move analytic to monographic when monographic is None
        assert ref.monographic_title == "Article Title"
        assert ref.analytic_title is None
    
    def test_empty_reference(self):
        """Test creating an empty Reference."""
        ref = Reference()
        
        assert ref.analytic_title is None
        assert ref.authors is None


class TestReferences:
    """Test References collection."""
    
    def test_create_references(self):
        """Test creating a References collection."""
        ref1 = Reference(monographic_title="Book One")
        ref2 = Reference(monographic_title="Book Two")
        
        refs = References([ref1, ref2])
        
        assert len(refs) == 2
        assert refs[0].monographic_title == "Book One"
        assert refs[1].monographic_title == "Book Two"
    
    def test_empty_references(self):
        """Test creating an empty References collection."""
        refs = References()
        
        assert len(refs) == 0 