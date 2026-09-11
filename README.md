# SSH Citation Index Modules

> **⚠️ Work in Progress**: This project is currently under active development. Features, APIs, and documentation are subject to change. Some components may be incomplete or experimental. Use in production environments is not recommended at this time.

## Description

SSH Citation Index modules are a collection of AI modules for extracting, parsing and disambiguating bibliographic references from publications in the Social Sciences and Humanities (SSH).

(Work on these modules is part of Odoma's contribution to deliverables [D4.2](https://operas.atlassian.net/wiki/spaces/GRAPHIA/pages/821329922/D4.2+-+Report+on+existing+datasets+methods+and+tools+for+the+SSH+Citation+Index) and [D4.4](https://operas.atlassian.net/wiki/spaces/GRAPHIA/pages/820838444/D4.4+-+Deployment+of+AI+modules+for+the+SSH+Citation+Index) in [WP4](https://operas.atlassian.net/wiki/spaces/GRAPHIA/pages/818087981/WP4+Artificial+Intelligence+Solutions+for+the+SSH+KG).)

## Installation

Current version: **v0.3.1**. See the [changelog](CHANGELOG.md) for release details.

For local Python development, use Python 3.10+:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

### Run the API locally

Copy `.env.example` to `.env` if you do not already have one. Set `LLM_ENDPOINT`
to your OpenAI-compatible vLLM endpoint, `LLM_API_KEY` to its credential, and
`LLM_MODEL_MEDIUM_INTELLIGENCE` to the served model name. Structured extraction
and parsing use Qwen3.6; keep `LLM_ENABLE_THINKING=false` for JSON output. The
endpoint must be reachable from the worker containers.

For MinerU, start the external service port-forward in a separate terminal:

```bash
oc port-forward service/mineru-api 8000:8000
```

Start the API, Redis, and workers from the repository root:

```bash
docker compose up -d --build --scale worker-llm=1 \
  redis api worker-default worker-llm worker-linking
```

The API is available on port **8001**; port **8000** is reserved for MinerU.
Containers use `MINERU_DOCKER_ENDPOINT`, defaulting to
`http://host.docker.internal:8000`. Outside Docker, configure `MINERU_ENDPOINT`
directly. MinerU runs as an external API service, not an imported Python library.

Check [service health](http://localhost:8001/health) and open the
[interactive API documentation](http://localhost:8001/docs). Health checks cover
Redis and storage; the later workflow stages exercise MinerU and the LLM.
See the [deployment guide](deployment/README.md) for OpenShift configuration.

## API workflow

Follow the [executable example notebook](examples/citation_index_api_guide.ipynb)
for the complete PDF-to-linked-reference workflow, polling helpers, and validation.

| Stage | Endpoint | Input |
|---|---|---|
| Extract text | `POST /extract/text` | Multipart PDF; extractor `pymupdf`, `grobid`, or `mineru` |
| Extract references | `POST /extract/references` | `{"text": "Markdown document text"}` |
| Parse references | `POST /parse/references` | `{"references": ["Raw citation string"]}` |
| Link references | `POST /link/references` | Parsed reference object or batch, shown below |

Each submission returns a `job_id`. Poll `GET /jobs/{job_id}/status` until
`completed` or `failed`, then retrieve completed results with
`GET /jobs/{job_id}`. Failed jobs expose an error in their status. The combined
`/process/references` route is not enabled.


## Tests

Run the isolated automated suite or just the linking checks:

```bash
pytest tests
pytest tests/test_citation_linking.py
```

The linking tests mock external services. The notebook makes real requests and
requires the API, workers, MinerU, and configured LLM to be available.

## Project Structure

```
citation_index/
├── src/citation_index/          # Core application code
│   ├── cli/                     # Command-line interface entry points
│   ├── core/                    # Domain logic and data models
│   │   ├── connectors/          # OpenAlex, OpenCitations, Wikidata, Matilda
│   │   ├── extractors/          # PDF extraction engines (Grobid, MinerU, PyMuPDF)
│   │   ├── models/              # Pydantic data models for references
│   │   ├── parsers/             # TEI-XML and bibliographic parsing
│   │   └── segmenters/          # Reference segmentation and localization
│   ├── llm/                     # LLM client bindings and prompt management
│   ├── pipelines/               # Extraction, parsing, and linking workflows
│   ├── api.py                   # FastAPI routes and request validation
│   ├── tasks.py                 # RQ background tasks
│   ├── evaluation/              # Metrics and evaluation scripts
│   └── utils/                   # Shared helper functions
├── tests/                       # Test suite mirroring src/ structure
├── benchmarks/                  # Evaluation datasets and scripts
│   ├── cex/                     # CEX benchmark dataset
│   ├── excite/                  # EXCITE dataset
│   ├── linkedbook/              # LinkedBooks dataset
│   ├── finetune/                # Fine-tuning datasets for LLM models
│   └── citation_linking/        # Citation linking scripts and test sets
├── prompts/                     # LLM prompt templates (YAML and Markdown)
├── examples/                    # API notebook and usage examples
└── deployment/                  # OpenShift manifests and deployment guide
```

## Current Deployment Status

### Text Extraction

- [x] Grobid integration
- [x] MinerU external API integration
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
- [x] Wikidata search connector
- [x] Matilda connector
- [x] Parsed-reference API with shared title/author/year matching
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
- [ ] CLI interface
- [x] Test suite
- [x] REST API module
- [x] API documentation and executable notebook
- [x] Deployment guides
- [x] Docker containerization

## Credits

The code contained in this repository is being developed by [Yurui Zhu](https://github.com/RuiaRui) ([Odoma](https://github.com/odoma-ch)). This work is carried out in the context of the EU-funded [GRAPHIA project](https://graphia-ssh.eu/) (grant ID: [101188018](https://cordis.europa.eu/project/id/101188018)).
