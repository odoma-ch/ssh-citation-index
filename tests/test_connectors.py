#!/usr/bin/env python3
"""Ad-hoc harness for exercising connector integrations."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

# Ensure the runtime package is importable when executed directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from citation_index.core.connectors import (
    MatildaConnector,
    OpenAlexConnector,
    OpenCitationsConnector,
    WikidataConnector,
)
from citation_index.core.models import Reference


class ConnectorTestSuite:
    """Lightweight smoke tests for each connector interface."""

    def __init__(self) -> None:
        self.results: Dict[str, Dict[str, Any]] = {}
        self.sample_refs: List[Reference] = [
            Reference(full_title="Attention Is All You Need"),
            Reference(full_title="Deep Learning"),
        ]

    def test_openalex(self) -> bool:
        name = "OpenAlex"
        print(f"\nTesting {name} connector…")

        try:
            connector = OpenAlexConnector()
            raw = connector.search(self.sample_refs[0], top_k=3)
            mapped = connector.map_to_references(raw)

            doi_rows = connector.search_by_id("10.48550/arXiv.1706.03762", "doi")
            doi_refs = connector.map_to_references(doi_rows)

            self.results[name] = {
                "status": "PASSED",
                "search_results": len(raw),
                "mapped_results": len(mapped.references),
                "doi_results": len(doi_refs.references),
            }
            print(f"  {name} connector: PASSED")
            return True
        except Exception as exc:  # pragma: no cover - diagnostic harness
            print(f"  {name} connector: FAILED - {exc}")
            self.results[name] = {"status": "FAILED", "error": str(exc)}
            return False

    def test_opencitations(self) -> bool:
        name = "OpenCitations"
        print(f"\nTesting {name} connector…")

        try:
            connector = OpenCitationsConnector()

            doi_rows = connector.search_by_id("10.1007/978-1-4020-9632-7", "doi")
            doi_refs = connector.map_to_references(doi_rows)

            title_raw = connector.search(self.sample_refs[0], top_k=3, threshold=50)
            title_refs = connector.map_to_references(title_raw)

            self.results[name] = {
                "status": "PASSED",
                "doi_results": len(doi_refs.references),
                "title_results": len(title_refs.references),
            }
            print(f"  {name} connector: PASSED")
            return True
        except Exception as exc:  # pragma: no cover - diagnostic harness
            print(f"  {name} connector: FAILED - {exc}")
            self.results[name] = {"status": "FAILED", "error": str(exc)}
            return False

    def test_matilda(self) -> bool:
        name = "Matilda"
        print(f"\nTesting {name} connector…")

        try:
            connector = MatildaConnector()

            query = Reference(full_title="Machine learning")
            raw = connector.search(query, top_k=3)
            mapped = connector.map_to_references(raw)

            doi_rows = connector.search_by_id("10.1007/978-1-4020-9632-7", "doi")
            doi_refs = connector.map_to_references(doi_rows)

            self.results[name] = {
                "status": "PASSED",
                "search_results": len(raw),
                "mapped_results": len(mapped.references),
                "doi_results": len(doi_refs.references),
            }
            print(f"  {name} connector: PASSED")
            return True
        except Exception as exc:  # pragma: no cover - diagnostic harness
            print(f"  {name} connector: FAILED - {exc}")
            self.results[name] = {"status": "FAILED", "error": str(exc)}
            return False

    def test_wikidata(self) -> bool:
        name = "Wikidata"
        print(f"\nTesting {name} connector…")

        try:
            connector = WikidataConnector()

            title_raw = connector.search(Reference(full_title="The Great Gatsby"), top_k=3)
            title_refs = connector.map_to_references(title_raw)

            sparql_raw = connector.search(
                Reference(full_title="The Great Gatsby"),
                top_k=3,
                method="sparql",
            )
            sparql_refs = connector.map_to_references(sparql_raw)

            doi_rows = connector.search_by_id("10.1007/978-1-4020-9632-7", "doi", top_k=1)
            doi_refs = connector.map_to_references(doi_rows)

            isbn_rows = connector.search_by_id("9780743273565", "isbn", top_k=3)
            isbn_refs = connector.map_to_references(isbn_rows)

            self.results[name] = {
                "status": "PASSED",
                "title_results": len(title_refs.references),
                "sparql_results": len(sparql_refs.references),
                "doi_results": len(doi_refs.references),
                "isbn_results": len(isbn_refs.references),
            }
            print(f"  {name} connector: PASSED")
            return True
        except Exception as exc:  # pragma: no cover - diagnostic harness
            print(f"  {name} connector: FAILED - {exc}")
            self.results[name] = {"status": "FAILED", "error": str(exc)}
            return False

    def run(self) -> bool:
        print("Running Citation Index connector checks")
        print("=" * 60)

        checks = [
            self.test_openalex,
            self.test_opencitations,
            self.test_matilda,
            self.test_wikidata,
        ]

        success = True
        for check in checks:
            success &= check()

        self._print_summary()
        return success

    def _print_summary(self) -> None:
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        for name, details in self.results.items():
            status = details.get("status", "UNKNOWN")
            print(f"{name:20} | {status}")
            for key, value in details.items():
                if key == "status":
                    continue
                print(f"{'':20} | {key.replace('_', ' ')}: {value}")
        print("=" * 60)


def main() -> None:
    suite = ConnectorTestSuite()
    ok = suite.run()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
