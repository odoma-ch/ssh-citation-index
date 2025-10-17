# API Connector Comparison

This document provides a comprehensive comparison of the available API connectors and their search capabilities.

## Search Capabilities Overview

| API | Title Search | Advanced Filtering | DOI/Identifier Search | SPARQL Search | Additional Features |
|-----|-------------|-------------------|---------------------|---------------|-------------------|
| **Matilda** | ✅ `query.title` | ✅ Author, Publisher, Date, Type, Sort | ✅ `query.identifier` (17 types: doi, isbn, arxiv, repec, pmid, eid, nlmuniqueid, rid, pii, pmcid, pmc, mid, bookaccession, versionId, version, medline, pmpid, hal) | ❌ | OrcId search, Description search |
| **OpenAlex** | ✅ `title.search` | ✅ Author(but need author id), Year, Type, Sort | ✅ DOI, ISBN, PMID, PMCID, ArXiv | ❌ | Citation counts, Open access info, Concepts |
| **Wikidata** | ✅ Elastic search | ✅ Author, Year, Type filtering | ✅ DOI, ISBN, PMID, ArXiv | Not Implemented | Entity relationships, Multilingual support |
| **OpenCitations** | ❌ | ❌ | ✅ DOI, PMID, PMCID | ✅ SPARQL queries |- |

## Detailed API Specifications

### Matilda Science API
- **Base URL**: `https://matilda.science/api`
- **Authentication**: Basic Auth (username/password)
- **Search Methods**:
  - Title search: `query.title`
  - Author search: `query.author`
  - Publisher search: `query.publisher`
  - Identifier search: `query.identifier`
  - OrcId search: `query.orcId`
  - Description search: `query.description`
- **Advanced Filtering**:
  - Date range: `filter.fromPublishedDate`, `filter.untilPublishedDate`
  - Work type: `filter.type`
  - Sorting: `sort` (score, created, updated, published, citedBy)
- **Supported Identifiers**: 17 types including DOI, ISBN, ArXiv, RePEc, PMID, etc.

### OpenAlex API
- **Base URL**: `https://api.openalex.org`
- **Authentication**: Optional Bearer token
- **Search Methods**:
  - Title search: `filter=title.search:{title}`
  - General search: `search` parameter
- **Advanced Filtering**:
  - Author: `filter=authorships.author.display_name.search:{author}`
  - Year: `filter=from_publication_date:{year}-01-01,to_publication_date:{year}-12-31`
  - Publisher: `filter=host_venue.display_name.search:{publisher}`
  - Work type: `filter=type:{type}`
  - Sorting: `sort=relevance_score:desc`
- **Supported Identifiers**: DOI, ISBN, PMID, PMCID, ArXiv ID

### Wikidata SPARQL
- **Base URL**: `https://query.wikidata.org/sparql`
- **Authentication**: None required
- **Search Methods**:
  - Elastic search: Uses Wikidata's built-in search
  - SPARQL queries: Custom SPARQL queries
- **Advanced Filtering**:
  - Author filtering: `wdt:P50` (author property)
  - Date filtering: `wdt:P577` (publication date)
  - Type filtering: `wdt:P31` (instance of)
- **Supported Identifiers**: DOI (`wdt:P356`), ISBN (`wdt:P212`), PMID (`wdt:P698`), ArXiv (`wdt:P818`)

### OpenCitations Meta API
- **Base URL**: `https://api.opencitations.net/meta/v1`
- **Authentication**: Optional access token
- **Search Methods**:
  - Title-based search with fuzzy matching
  - SPARQL queries for complex searches
- **Advanced Filtering**:
  - Author matching with similarity scoring
  - Year-based filtering
  - Publisher matching
- **Supported Identifiers**: DOI, PMID, PMCID
- **Special Features**: Citation network analysis, reference matching algorithms

