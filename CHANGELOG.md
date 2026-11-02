# Changelog

All notable changes to this project are documented here.

## [0.4.2] - 2026-11-02

### Fixed
- GROBID page ranges were dropped from `parsed_references`. GROBID writes them as `<biblScope unit="page" from="221" to="234"/>` with no text, and `TeiBiblParser` read only the text. It now falls back to `from`/`to` (`"221-234"`, or one page when only `from` is set or both are equal); volume and issue use the same path. On six test papers 154 page ranges were lost before; field-by-field comparison with the raw TEI now matches on every field. The CLI and benchmarks share this parser, so gold TEI page ranges are read too.

### Added
- `api_0.4.1.yml` regenerated as `api_0.4.2.yml` (version only).
- Tests for the 0.4.1 request validation (`tests/test_api_validation.py`) and for per-source linking errors.

### Known issues
- Citation linking: the match rule accepts title >= 90 plus author or year, and the title score counts containment as 100. Reviews of a book and duplicate records can be linked instead of the work, and a different edition is linked without a flag. About 61 s per reference with `target=all` when the OpenCitations SPARQL fallback times out.
- MinerU (`vlm-auto-engine`) drops footnotes and garbles Greek on footnote-style humanities papers; prefer PyMuPDF there.
- The LLM parser copies the page span into `cited_range` as well as `pages`.

## [0.4.1] - 2026-11-02

### Fixed
- `POST /extract/text` (PyMuPDF) failed with `No module named 'pymupdf4llm'`: `pymupdf4llm>=0.0.27` resolved to 2.0, an empty shim for `pymupdf>=2` (which needs Python >= 3.11; the image runs 3.10). Pinned `pymupdf4llm<2.0`.
- `requirements.txt` was ignored by the `*.txt` rule in `.gitignore` and never committed; it is now tracked.
- Citation linking: one failing source (e.g. an OpenAlex 504) no longer fails the whole job. Its link is returned as `{"id": null, "doi": null, "error": "..."}` and the other sources keep their links. The job still fails when every lookup fails.
- `prompt_name` on `/extract/references` and `/parse/references` must now be an existing file under `prompts/` (400 otherwise). Before, the worker opened any path, so a caller could send server files such as `/proc/self/environ` to the LLM.

### Changed
- Unknown query values are rejected with 422 before a job is queued: `parser` (`llm`, `grobid`), `extractor` (`pymupdf`, `mineru`, `grobid`), `method` (`full_text`, `semantic`, `grobid`), and `temperature` outside 0–2. Unknown `parser` / `method` values used to run silently as the LLM / `full_text` path; `page_by_page` was documented but never implemented and is no longer listed.
- `api_0.4.0.yml` regenerated as `api_0.4.1.yml`.

## [0.4.0] - 2026-11-02

### Added
- `POST /classify/citations`: queues citation-intent classification (method / background / result) with the external CEC classifier (`CEC_ENDPOINT`, `CEC_TIMEOUT`; `mode` is `WS`, `WoS` or `M`). Runs on the `default` queue with its own `timeout_cec_classification`. CEC serves one request at a time and rebuilds its predictor per call, so send a paper's citations in one request.
- `POST /process/references/grobid`: PDF to raw and parsed references with GROBID fulltext. `include_context=true` adds, per reference, the citing sentences as `{section, citation, marker, ref_id}`; these can be posted as-is to `/classify/citations`.
- `POST /extract/references` accepts a multipart PDF with `method=grobid` (the default for uploads), plus `include_parsed` and `include_context`. Results stay index-aligned: `references` (raw strings), `parsed_references`, `contexts`. The JSON `text` body (LLM) is unchanged; mismatched combinations return 400.
- OpenCitations linking target (`target=opencitations`, also part of `all`). The connector looks up the reference's DOI through the Meta REST API, then falls back to a SPARQL title-prefix / family-name + year search on `sparql.opencitations.net/meta`. Candidate IDs are OMIDs (`br/...`). Optional `OPENCITATIONS_API` (sent as the `authorization` header; wired into `worker-linking` in compose and K8s).

### Fixed
- `GrobidClient.process_full_text` sent `consolidateReferences` / `includeRawReferences`, which GROBID ignores; it now sends `consolidateCitations` / `includeRawCitations`. Raw reference strings are now actually returned and the `consolidate_references` flag now takes effect. `process_references` still uses the old names.

### Changed
- Version bumped to 0.4.0 (package, image tags in `deployment/`). `TIMEOUT_CEC_CLASSIFICATION` is now set in `docker-compose.yml`, the ConfigMap and `.env.example`.
- `api_0.0.1.yml` (hand-written, listed endpoints that do not exist) replaced by `api_0.4.0.yml`, generated from the FastAPI app; it includes `/link/references`, `/classify/citations`, `/process/references/grobid` and the `X-API-Key` scheme.
- `TIMEOUT_CITATION_LINKING` default raised from 600 to 3600 seconds: the OpenCitations SPARQL fallback averages ~6s per reference and can take minutes for common surnames.
- OpenCitations connector failures raise `OpenCitationsError` instead of returning an empty list; the connector no longer needs SPARQLWrapper.

## [0.3.1] - 2026-09-11

### Changed
- Citation linking accepts parsed reference objects with required `full_title`, `authors`, and `publication_year` fields. Batches use a JSON array with `batched=true`; raw reference strings are no longer accepted.
- Linking searches with parsed fields and filters candidates with the shared `custom_match()` function in the main package: title similarity at least 90 plus either author similarity at least 70 or year within one year.
- The API example notebook links the parsing stage's output and validates the v0.3.1 request contract.

## [0.3.0] - 2026-08-28

Covers `d63858b..e204f40` (2026-08-27).

### Added

- Queue-backed `POST /link/references` API for one reference or a
  newline-delimited batch.
- Citation linking against OpenAlex, Matilda, Wikidata, `all`, or a
  comma-separated target list, returning each target's ID and DOI when found.
- Dedicated `linking` RQ worker and an enabled OKD worker deployment.
- End-to-end API/queue tests and executable notebook examples for citation
  linking.

### Changed

- Connector request failures are logged with tracebacks and propagated so RQ
  marks failed linking jobs as failed; identifier lookup 404s remain normal
  no-match results.
- Matilda credentials are passed to the linking worker.

## [0.2.2] - 2026-08-18

Covers `a963a43..d63858b` (2026-07-22 → 2026-08-14). No git tags exist for
earlier releases; the `0.1.0 → 0.2.0` version bump falls inside this range, and
`0.2.1` was never cut, so everything below is released together as `0.2.2`.

### Removed

- **Breaking:** the Marker extractor is gone — removed from `ExtractorFactory`,
  the extractor package exports, and the CLI `--extractor` choices. Callers
  passing `marker` must switch to `pymupdf`, `mineru`, or `grobid`.

### Added

- Kubernetes deployment for the MinerU service (`deployment/mineru.yaml`,
  `deployment/mineru.Dockerfile`).
- `MINERU_ENDPOINT`, `MINERU_TIMEOUT`, and `MINERU_BACKEND` settings
  (default backend `vlm-auto-engine`).
- Per-stage queue timeouts: `TIMEOUT_TEXT_EXTRACTION`,
  `TIMEOUT_REFERENCE_EXTRACTION`, `TIMEOUT_REFERENCE_PARSING`,
  `TIMEOUT_CITATION_LINKING`, validated against the LLM timeout × retries at
  startup.
- Per-stage first-token timeouts for LLM streaming calls, plus separate
  connect / write / pool timeouts on the HTTP client.
- `LLM_ENABLE_THINKING` (default `false`), plumbed to vLLM chat-template
  arguments so structured output lands in the answer `content` rather than the
  reasoning channel.
- `REDIS_PASSWORD` setting for authenticated Redis instances.
- Tests: `test_mineru_extractor`, `test_reference_schema`,
  `test_reference_parsing_errors`, `test_reference_pipeline_live`.

### Changed

- MinerU is now consumed as an external HTTP service: the extractor uploads to
  `POST /file_parse` and reads `md_content` only. The MinerU Python package is
  no longer an application dependency.
- Reference parsing now requires concrete fields in the guided-decoding schema
  (`Reference.schema_without_excluded()`). With nothing marked required, Qwen
  silently dropped whole fields — 0 of 63 titles emitted on a footnote-style
  PDF; requiring only the titles then dropped every author. `raw` and
  `identifiers` stay optional on purpose: requiring them made the model emit a
  malformed `raw` string that broke JSON parsing outright.
- Docker Compose publishes the API on host port `8001`, leaving `8000` free for
  the MinerU port-forward; `host.docker.internal` is used to reach it.
- `LLM_API_KEY` is read from the environment only — no key value is present in
  tracked config.

### Fixed

- Structured output: a JSON schema passed to `LLMClient.call()` is now wrapped
  as `{"type": "json_schema", "json_schema": {"name": ..., "schema": ...}}`.
  The previous raw-schema form was accepted by the SDK but reached vLLM as an
  empty constraint, so responses were unconstrained.
- Extraction, parsing, and MinerU failures now log with context, record the
  error in job metadata, and re-raise. Previously an LLM or API failure could
  surface as a successful job with an empty reference list. MinerU failures
  raise `MineruAPIError` instead of returning empty Markdown.
- Default medium-intelligence model name corrected to `Qwen3.6-27B`.

## [0.2.0] - 2026-07-22

Initial queue-backed release (API + RQ workers + filesystem storage). Not
tagged in git; shipped as container image `citation-index:0.2.0`.
