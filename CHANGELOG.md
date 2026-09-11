# Changelog

All notable changes to this project are documented here.

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
