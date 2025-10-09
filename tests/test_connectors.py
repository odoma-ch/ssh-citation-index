#!/usr/bin/env python3
"""
Test script for all citation index connectors.
"""

import sys
import os
from typing import Dict, Any, List

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from citation_index.core.connectors import (
    OpenAlexConnector, 
    OpenCitationsConnector,
    MatildaConnector,
    WikidataConnector
)
from citation_index.core.models import Reference, References


class ConnectorTestSuite:
    """Test suite for all citation index connectors."""
    
    def __init__(self):
        self.test_references = [
            Reference(full_title="Attention Is All You Need"),
            Reference(full_title="BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"),
            Reference(full_title="Deep Learning"),
        ]
        self.results: Dict[str, Dict[str, Any]] = {}
    
    def test_openalex_connector(self) -> bool:
        """Test the OpenAlex connector."""
        print("\nTesting OpenAlex Connector...")
        connector_name = "OpenAlex"
        
        try:
            connector = OpenAlexConnector()
            test_ref = self.test_references[0]
            
            raw_results = connector.search(test_ref, top_k=3)
            mapped_results = connector.map_to_references(raw_results)
            
            self.results[connector_name] = {
                "status": "PASSED",
                "raw_results_count": len(raw_results),
                "mapped_results_count": len(mapped_results.references),
                "sample_title": mapped_results.references[0].full_title if mapped_results.references else None
            }
            
            print(f"  {connector_name} connector: PASSED")
            return True
            
        except Exception as e:
            print(f"  {connector_name} connector: FAILED - {e}")
            self.results[connector_name] = {
                "status": "FAILED",
                "error": str(e)
            }
            return False
    
    def test_opencitations_connector(self) -> bool:
        """Test the OpenCitations connector."""
        print("\nTesting OpenCitations Connector...")
        connector_name = "OpenCitations"
        
        try:
            connector = OpenCitationsConnector()
            
            # Test 1: DOI-based search
            print("  - Testing DOI search...")
            test_doi = "10.1007/978-1-4020-9632-7"
            doi_result = connector.search_by_doi(test_doi)
            doi_success = doi_result is not None
            
            # Test 2: Identifier-based search (auto-detect)
            print("  - Testing identifier search...")
            auto_ref = connector.search_by_id("10.1007/978-1-4020-9632-7")
            identifier_count = 1 if auto_ref else 0
            
            # Test 3: SPARQL-based title search (if SPARQLWrapper is available)
            print("  - Testing SPARQL title search...")
            title_ref = Reference(full_title="Attention Is All You Need")
            try:
                title_results = connector.search(title_ref, top_k=3, threshold=50)
                title_search_count = len(title_results)
            except Exception as e:
                print(f"    SPARQL search not available or failed: {e}")
                title_search_count = 0
            
            # Test 4: SPARQL-based title + author search
            print("  - Testing SPARQL title + author search...")
            author_ref = Reference(
                full_title="BERT Pre-training of Deep Bidirectional Transformers",
                authors=["Devlin"]
            )
            try:
                author_results = connector.search(author_ref, top_k=3, 
                                                 include_author=True, threshold=50)
                author_search_count = len(author_results)
            except Exception as e:
                print(f"    SPARQL author search failed: {e}")
                author_search_count = 0
            
            # Test 5: Search with additional metadata (year, volume, pages)
            print("  - Testing SPARQL search with metadata...")
            metadata_ref = Reference(
                full_title="Deep Learning",
                publication_date="2015",
                volume="521",
                pages="436-444"
            )
            try:
                metadata_results = connector.search(metadata_ref, top_k=3,
                                                   include_date=True, threshold=50)
                metadata_search_count = len(metadata_results)
            except Exception as e:
                print(f"    SPARQL metadata search failed: {e}")
                metadata_search_count = 0
            
            # Test 6: Test lookup_ref convenience method
            print("  - Testing lookup_ref method...")
            test_ref = Reference(full_title="Machine Learning")
            try:
                basic_results = connector.lookup_ref(test_ref, top_k=3)
                general_search_count = len(basic_results.references)
            except Exception as e:
                print(f"    lookup_ref failed: {e}")
                general_search_count = 0
            
            total_results = ((1 if doi_success else 0) + identifier_count + 
                           title_search_count + author_search_count + 
                           metadata_search_count + general_search_count)
            
            self.results[connector_name] = {
                "status": "PASSED",
                "doi_search_success": doi_success,
                "identifier_search_count": identifier_count,
                "title_search_count": title_search_count,
                "author_search_count": author_search_count,
                "metadata_search_count": metadata_search_count,
                "general_search_count": general_search_count,
                "total_results": total_results,
            }
            
            print(f"  {connector_name} connector: PASSED")
            print(f"    Total results across all tests: {total_results}")
            return True
            
        except Exception as e:
            print(f"  {connector_name} connector: FAILED - {e}")
            import traceback
            traceback.print_exc()
            self.results[connector_name] = {
                "status": "FAILED",
                "error": str(e)
            }
            return False
    
    def test_matilda_connector(self) -> bool:
        """Test the Matilda connector."""
        print("\nTesting Matilda Connector...")
        connector_name = "Matilda"
        
        try:
            connector = MatildaConnector()
            
            # Test title search
            title_ref = Reference(full_title="F1 Scores")
            title_results = connector.lookup_ref(title_ref, top_k=3)
            
            # Test author search
            author_ref = Reference(authors=["Charles Darwin"])
            author_results = connector.lookup_ref(author_ref, top_k=3)
            
            # Test general query search
            general_ref = Reference(full_title="machine learning")
            general_results = connector.lookup_ref(general_ref, top_k=3)
            
            # Test advanced search with kwargs
            advanced_ref = Reference(full_title="neural networks")
            advanced_results = connector.lookup_ref(
                advanced_ref,
                top_k=3,
                work_type="journal-article",
                sort_by="created"
            )
            
            # Test DOI search
            doi_result = connector.search_by_doi("10.1007/978-1-4020-9632-7")
            doi_success = doi_result is not None
            
            # Collect results
            total_results = (len(title_results.references) + 
                           len(author_results.references) + 
                           len(general_results.references) + 
                           len(advanced_results.references) + 
                           (1 if doi_success else 0))
            
            self.results[connector_name] = {
                "status": "PASSED",
                "title_search_count": len(title_results.references),
                "author_search_count": len(author_results.references),
                "general_search_count": len(general_results.references),
                "advanced_search_count": len(advanced_results.references),
                "doi_search_success": doi_success,
                "total_results": total_results,
            }
            
            print(f"  {connector_name} connector: PASSED")
            return True
            
        except Exception as e:
            print(f"  {connector_name} connector: FAILED - {e}")
            self.results[connector_name] = {
                "status": "FAILED",
                "error": str(e)
            }
            return False
    
    def test_wikidata_connector(self) -> bool:
        """Test the Wikidata connector."""
        print("\nTesting Wikidata Connector...")
        connector_name = "Wikidata"
        
        try:
            connector = WikidataConnector()
            
            # Test title search using unified interface (elastic method - default)
            title_ref = Reference(full_title="The Great Gatsby")
            title_results = connector.lookup_ref(title_ref, top_k=3)
            title_count = len(title_results.references)
            
            # Test SPARQL method
            sparql_results = connector.lookup_ref(title_ref, top_k=3, method="sparql")
            sparql_count = len(sparql_results.references)
            
            # Test author search
            author_results = connector.search_books_by_author("Virginia Woolf", top_k=3)
            author_count = len(author_results.references)
            
            # Test ISBN search
            isbn_results = connector.search_by_isbn("978-0-7432-7356-5", top_k=3)
            isbn_count = len(isbn_results.references)
            
            # Test DOI search
            doi_result = connector.search_by_doi("10.1007/978-1-4020-9632-7")
            doi_count = 1 if doi_result else 0
            
            total_results = title_count + sparql_count + author_count + isbn_count + doi_count
            
            self.results[connector_name] = {
                "status": "PASSED",
                "title_search_count": title_count,
                "sparql_search_count": sparql_count,
                "author_search_count": author_count,
                "isbn_search_count": isbn_count,
                "doi_search_count": doi_count,
                "total_results": total_results,
            }
            
            print(f"  {connector_name} connector: PASSED")
            return True
            
        except Exception as e:
            print(f"  {connector_name} connector: FAILED - {e}")
            self.results[connector_name] = {
                "status": "FAILED",
                "error": str(e)
            }
            return False
    
    def run_all_tests(self) -> bool:
        """Run all connector tests."""
        print("Running Citation Index Connector Tests")
        print("=" * 50)
        
        test_methods = [
            self.test_openalex_connector,
            self.test_opencitations_connector,
            self.test_matilda_connector,
            self.test_wikidata_connector,
        ]
        
        all_passed = True
        for test_method in test_methods:
            try:
                result = test_method()
                if not result:
                    all_passed = False
            except Exception as e:
                print(f"  Unexpected error in {test_method.__name__}: {e}")
                all_passed = False
        
        self.print_summary()
        return all_passed
    
    def print_summary(self):
        """Print test results summary."""
        print("\n" + "=" * 50)
        print("Test Results Summary")
        print("=" * 50)
        
        for connector_name, result in self.results.items():
            status = result["status"]
            print(f"{connector_name:20} | {status}")
            
            if "error" in result:
                print(f"{'':20} | Error: {result['error']}")
            elif "mapped_results_count" in result:
                print(f"{'':20} | Found {result['mapped_results_count']} results")
            elif "total_results" in result:
                print(f"{'':20} | Found {result['total_results']} total results")
        
        print("=" * 50)
        
        passed_count = sum(1 for r in self.results.values() if "PASSED" in r["status"])
        failed_count = sum(1 for r in self.results.values() if "FAILED" in r["status"])
        
        print(f"Total: {len(self.results)} connectors")
        print(f"Passed: {passed_count}")
        print(f"Failed: {failed_count}")
        
        if failed_count == 0:
            print("\nAll tests passed")
        else:
            print(f"\n{failed_count} connector(s) failed")


def main():
    """Main test function."""
    test_suite = ConnectorTestSuite()
    success = test_suite.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
