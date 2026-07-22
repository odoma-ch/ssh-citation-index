#!/usr/bin/env python3
"""
Tests for GROBID client functionality.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

# Add the src directory to the path so we can import citation_index
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from citation_index.llm.grobid_client import GrobidClient, GrobidError

class TestGrobidCitationParsing:
    """Tests for GROBID citation parsing functionality."""
    
    @pytest.fixture
    def grobid_client(self):
        """Create a GROBID client for testing."""
        return GrobidClient(endpoint="https://grobid-graphia-app1-staging.apps.bst2.paas.psnc.pl/", timeout=30.0, max_retries=2)
    
    @pytest.fixture
    def sample_citation(self):
        """Sample citation for testing."""
        return "Graff, Expert. Opin. Ther. Targets (2002) 6(1): 103-113"
    
    @pytest.fixture
    def sample_citations_list(self):
        """Sample list of citations for testing."""
        return [
            "Smith, J. (2020). Title of Article. Journal Name, 10(2), 45-67.",
            "Doe, A. (2019). Book Title. Publisher.",
            "Brown, C., & Green, D. (2021). Another Article. Science, 123, 456-789."
        ]
    
    @pytest.fixture
    def sample_tei_xml_response(self):
        """Sample TEI XML response from GROBID."""
        return """<?xml version="1.0" encoding="UTF-8"?>
<biblStruct>
    <analytic>
        <title/>
        <author>
            <persName xmlns="http://www.tei-c.org/ns/1.0"><surname>Graff</surname></persName>
        </author>
    </analytic>
    <monogr>
        <title level="j">Expert. Opin. Ther. Targets</title>
        <imprint>
            <biblScope unit="volume">6</biblScope>
            <biblScope unit="issue">1</biblScope>
            <biblScope unit="page" from="103" to="113" />
            <date type="published" when="2002" />
        </imprint>
    </monogr>
</biblStruct>"""
    
    def test_process_citation_list_single_citation(self, grobid_client, sample_citation, sample_tei_xml_response):
        """Test parsing single citation."""
        with patch.object(grobid_client.session, 'post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = sample_tei_xml_response
            mock_post.return_value = mock_response
            
            result = grobid_client.process_citation_list([sample_citation])
            
            assert result == sample_tei_xml_response
            assert mock_post.called
    
    def test_process_citation_list_with_raw_citations(self, grobid_client, sample_citation, sample_tei_xml_response):
        """Test parsing with raw citations included."""
        with patch.object(grobid_client.session, 'post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = sample_tei_xml_response
            mock_post.return_value = mock_response
            
            result = grobid_client.process_citation_list(
                [sample_citation],
                include_raw_citations=True
            )
            
            assert result == sample_tei_xml_response
            assert mock_post.called
    
    def test_error_204_no_content(self, grobid_client, sample_citation):
        """Test 204 No Content error."""
        with patch.object(grobid_client.session, 'post') as mock_post:
            mock_post.return_value = Mock(status_code=204, text="")
            
            with pytest.raises(GrobidError, match="HTTP 204"):
                grobid_client.process_citation_list([sample_citation])
    
    def test_error_400_bad_request(self, grobid_client, sample_citation):
        """Test 400 Bad Request error."""
        with patch.object(grobid_client.session, 'post') as mock_post:
            mock_post.return_value = Mock(status_code=400, text="Invalid")
            
            with pytest.raises(GrobidError, match="HTTP 400"):
                grobid_client.process_citation_list([sample_citation])
    
    def test_error_500_server_error(self, grobid_client, sample_citation):
        """Test 500 Server Error."""
        with patch.object(grobid_client.session, 'post') as mock_post:
            mock_post.return_value = Mock(status_code=500, text="Error")
            
            with pytest.raises(GrobidError, match="HTTP 500"):
                grobid_client.process_citation_list([sample_citation])
    
    def test_503_retry_then_success(self, grobid_client, sample_citation, sample_tei_xml_response):
        """Test 503 triggers retry and then succeeds."""
        with patch.object(grobid_client.session, 'post') as mock_post, \
             patch('time.sleep'):
            mock_post.side_effect = [
                Mock(status_code=503, text="Unavailable"),
                Mock(status_code=200, text=sample_tei_xml_response)
            ]
            
            result = grobid_client.process_citation_list([sample_citation])
            
            assert result == sample_tei_xml_response
            assert mock_post.call_count == 2
    
    def test_batch_multiple_citations(self, grobid_client, sample_citations_list, sample_tei_xml_response):
        """Test parsing multiple citations."""
        with patch.object(grobid_client.session, 'post') as mock_post:
            mock_post.return_value = Mock(status_code=200, text=sample_tei_xml_response)
            
            result = grobid_client.process_citation_list(sample_citations_list)
            
            assert result == sample_tei_xml_response
            assert mock_post.called
    
    def test_empty_list_raises_error(self, grobid_client):
        """Test empty list raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            grobid_client.process_citation_list([])
    
    def test_all_empty_strings_raises_error(self, grobid_client):
        """Test all empty strings raises ValueError."""
        with pytest.raises(ValueError, match="no valid citation"):
            grobid_client.process_citation_list(["", "  ", "\n"])
    
    def test_filters_empty_strings(self, grobid_client, sample_tei_xml_response):
        """Test empty strings are filtered out."""
        with patch.object(grobid_client.session, 'post') as mock_post:
            mock_post.return_value = Mock(status_code=200, text=sample_tei_xml_response)
            
            citations = ["Smith, J. (2020).", "", "Doe, A. (2019).", "  "]
            result = grobid_client.process_citation_list(citations)
            
            assert result == sample_tei_xml_response
    
    def test_empty_response_raises_error(self, grobid_client, sample_citation):
        """Test empty response raises error."""
        with patch.object(grobid_client.session, 'post') as mock_post, patch(
            "time.sleep"
        ):
            mock_post.return_value = Mock(status_code=200, text="   ")
            
            with pytest.raises(GrobidError, match="empty response"):
                grobid_client.process_citation_list([sample_citation])
    
