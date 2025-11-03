# SSH Citation Index Modules

> **⚠️ Work in Progress**: This project is currently under active development. Features, APIs, and documentation are subject to change. Some components may be incomplete or experimental. Use in production environments is not recommended at this time.

## Description

SSH Citation Index modules are a collection of AI modules for extracting, parsing and disambiguating bibliografic references from publications in the Social Sciences and Humanities (SSH). 

(Work on these modules is part of Odoma's contribution to deliverables [D4.2](https://operas.atlassian.net/wiki/spaces/GRAPHIA/pages/821329922/D4.2+-+Report+on+existing+datasets+methods+and+tools+for+the+SSH+Citation+Index) and [D4.4](https://operas.atlassian.net/wiki/spaces/GRAPHIA/pages/820838444/D4.4+-+Deployment+of+AI+modules+for+the+SSH+Citation+Index) in [WP4](https://operas.atlassian.net/wiki/spaces/GRAPHIA/pages/818087981/WP4+Artificial+Intelligence+Solutions+for+the+SSH+KG).)


## Installation

Installation instructions are a work in progress. The project requires Python 3.10+ and dependencies listed in `requirements.txt`.

## Project Structure

```
citation_index/
├── src/citation_index/          # Core application code
│   ├── cli/                     # Command-line interface entry points
│   ├── core/                    # Domain logic and data models
│   │   ├── connectors/          # External API integrations for citation linking(OpenAlex, OpenCitations, Wikidata, Matilda)
│   │   ├── extractors/          # PDF extraction engines (Grobid, Marker, MinerU, PyMuPDF)
│   │   ├── models/              # Pydantic data models for references
│   │   ├── parsers/             # TEI-XML and bibliographic parsing
│   │   └── segmenters/          # Reference segmentation and localization
│   ├── llm/                     # LLM client bindings and prompt management
│   ├── pipelines/               # Extraction and parsing workflow orchestration
│   ├── evaluation/              # Metrics and evaluation scripts
│   └── utils/                   # Shared helper functions
├── tests/                       # Test suite mirroring src/ structure
├── benchmarks/                  # Evaluation datasets and scripts
│   ├── cex/                     # CEX benchmark dataset
│   ├── excite/                  # EXCITE dataset
│   ├── linkedbook/              # LinkedBooks dataset
│   └── finetune/                # Fine-tuning datasets for LLM models
├── prompts/                     # LLM prompt templates (YAML and Markdown)
└── scripts/                     # Utility scripts for data processing and benchmarking
```

### Key Components

- **Extractors**: Integrate with multiple PDF parsing engines (Grobid, Marker, MinerU, PyMuPDF) to extract reference sections from academic papers
- **Parsers**: Parse extracted references into structured data models using TEI-XML parsing or LLM-based approaches
- **Connectors**: Link parsed references to external knowledge bases (OpenAlex, OpenCitations, Wikidata, Matilda) for disambiguation
- **Pipelines**: Orchestrate end-to-end workflows from PDF to parsed and disambiguated references
- **CLI**: Command-line tools for running extraction, parsing, and evaluation tasks

## Current Deployment Status

### Text Extraction 
- [x] Grobid integration
- [x] Marker PDF integration
- [x] MinerU integration
- [x] PyMuPDF integration
- [x] Extractor comparison and benchmarking

### Reference Extraction and Parsing
- [x] TEI-XML parser (Grobid output)
- [x] LLM-based parser
- [x] Prompt templates and variants
- [x] Semantic reference locator/segmenter
- [x] Benchmarking: EXCITE, CEXgoldstandard, LinkedBooks

### Citation Linking
- [x] OpenAlex API connector
- [x] OpenCitations API connector
- [x] Wikidata SPARQL connector
- [x] Matilda connector
- [x] Simple search and match pipeline
- [ ] Advanced search and match pipeline
- [ ] benchmark datasets(cex, excite, linkedbooks)
  - [x] creation 
  - [ ] annotation
  - [ ] evaluation

### Citation Intent Classification
- TODO

### Entity Extraction (software, dataset, funding, entity mentions)
- TODO

### Infrastructure
- [x] Core data models (Reference, Person, Organization)
- [x] LLM client with retry logic
- [x] CLI interface
- [x] Test suite
- [ ] REST API module
- [ ] API documentation
- [ ] Deployment guides
- [ ] Docker containerization





## Credits

The code contained in this repository is being developed by [Yurui Zhu](https://github.com/RuiaRui) ([Odoma](https://github.com/odoma-ch)). This work is carried out in the context of the EU-funded [GRAPHIA project](https://graphia-ssh.eu/) (grant ID: [101188018](https://cordis.europa.eu/project/id/101188018)).