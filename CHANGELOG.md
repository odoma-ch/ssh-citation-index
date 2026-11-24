# Changelog

All notable changes to this project are documented here.

## [0.5.2] - 2026-11-24

### Added
- `Reference.container_title` (#54): the book that contains a chapter, or the proceedings that contain a conference paper. `journal_title` keeps journals and other serials. The field is in the API output and, as a required key, in the guided-decoding schema, placed before `journal_title` so the model decides on a container first. The parsing prompts describe it and show a book-chapter example.
- GROBID / TEI parsing sets `ref_type` (`journal-article`, `book-chapter`, `book`, `proceedings-paper`, else none) and `container_title` from the `<biblStruct>` structure. Only a `<meeting>` with its own text counts as a conference: GROBID files most places of publication as `<meeting><address>`. 

- `POST /extract/metadata?method=grobid|llm` (#54): bibliographic metadata of the uploaded PDF itself (the citing work), as one `Reference`. `grobid` reads the fulltext TEI header (`default` queue); `llm` reads the first two pages with the new `metadata.default` prompt (`llm-tasks` queue). GROBID's `md5` file hash is dropped from the identifiers.
- GROBID PDF responses are cached under `STORAGE_ROOT/grobid_cache/{pdf_sha256}/`, keyed by service and params, so the same PDF sent to another GROBID endpoint skips GROBID. Only successful responses are cached; the storage cleanup CronJob prunes the cache like job folders. Clear it after a GROBID upgrade.

### Changed
- GROBID reference extraction always asks GROBID for sentence segmentation, so references, contexts and metadata share one cached TEI.
- Reference extraction prompts (`reference_extraction.md`, `end_to_end_parsing.md`, `extraction.default`, `extraction_and_parsing.default`) say not to extract the citation of the document itself.
- Parsing prompt few-shots: the book-chapter author has a middle initial (`Lee, K. J.`), which keeps Qwen splitting initials into `first_name` / `middle_name` once `container_title` is in the schema; "UN Publications" is a `publisher`, no longer a `journal_title`.

### Fixed
- DOI normalisation strips trailing `.`, `,` and `;` picked up from PDF text (`10.1037/apl0000399.`).
- TEI export without `raw.tei` writes a journal article's title as `level="a"`, not as a monograph title.

## [0.5.1] - 2026-11-19

### Added
- `POST /link/references?filter_reviews_by_doi=true` drops candidates whose DOI only ever names a book review (Choice Reviews Online, `10.5860/choice.*`) before `match` or `model` chooses. Default `false`; results record the option.

### Changed
- Linking decider weights `v5_select_v1` (Hub revision `09135525`, SHA-256 `76520c56…`, none-logit bias −3.217), retrained on the reviewed citation-linking dataset (revision `ed992b5`). Held-out test F1 84.5% (P 90.1%, R 79.5%). The service defaults and `deployment/linking-decider.yaml` carry the new SHA and bias.
- `deployment/linking-decider.yaml` loads the weights from the shared PVC (`citation-index-storage`, `models/linking-decider/v5_select_v1`, via `DECIDER_MODEL_PATH`) instead of the Hub: no `HF_TOKEN` needed, no 1.1 GB download per restart. Explicit ephemeral-storage requests/limits for the namespace quota. Copy the checkpoint to the PVC before the first rollout (command in the manifest header).
- Deploy order as for 0.5.0: decider, config map (adds `LINKING_DECIDER_ENDPOINT`), `worker-linking`, then the API and the other workers.
- Citation-linking dataset: every label checked against the frozen candidate pool (`CORRECTIONS.md`, round 6), duplicate candidates removed, corrections moved to `benchmarks/citation_linking/dataset/corrections.py`, a QA report written at build time.

## [0.5.0] - 2026-11-18

### Added
- `POST /link/references?method=model` (#58): the fine-tuned GLiNER2 decider (`yurui983/gliner2.5-multi-decide-citation-linking`, none-logit bias +7.737) chooses among each index's search results, or none. References need `raw_reference`. Links gain a `score`; results and job metadata record `method`. The default `method=match` keeps the title/author/year rules.
- Linking decider service (`services/linking_decider`, `deployment/linking-decider.Dockerfile`, `deployment/linking-decider.yaml`, compose `linking-decider`). CPU-only torch; refuses weights whose SHA-256 differs from the validated checkpoint. Model input formatting is copied from the training export and checked against exported examples. New settings `LINKING_DECIDER_ENDPOINT` and `LINKING_DECIDER_TIMEOUT`; the K8s secret needs `HF_TOKEN`.
- Deploy order: roll out `worker-linking` before the API. An older worker ignores the stored `method` and would run `match` for a `method=model` job.
- A decider failure fails the linking job instead of being recorded per index (it is not one source among several).

### Changed
- Version bumped to 0.5.0 (package, image tags; the decider image is tagged `linking-decider:0.5.0` alongside the app). OpenAPI spec regenerated as `api_0.5.0.yml`.

## [0.4.3] - 2026-11-18

### Changed
- Docker image now uses Python 3.12 (`python_requires>=3.11`) and PyMuPDF 2.x. The PyMuPDF extractor calls `Document.to_markdown()`; the separate `pymupdf4llm` pin is gone (PyMuPDF 2 installs a matching version, plus `pymupdf_layout` and `onnxruntime`, so the image is larger). (#56)
- Job retention aligned (#57): `WORKER_RESULT_TTL` default 3600 → 86400, new `WORKER_FAILURE_TTL` (default 604800) passed as `failure_ttl` on every task decorator and `enqueue_call` (failed RQ records previously kept for a year). Config defaults now match production: `JOB_METADATA_TTL=86400`, `STORAGE_CLEANUP_AGE_HOURS=72`. See "Job retention" in the README.

### Fixed
- `GET /jobs/{id}/status` no longer calls `Job.fetch` for completed or failed jobs, so it stops logging `Could not fetch RQ job ...` on every poll once the RQ record has expired.

## [0.4.2] - 2026-11-02

### Added
- `POST /classify/citations`: queues citation-intent classification (method / background / result) with the external CEC classifier (`CEC_ENDPOINT`, `CEC_TIMEOUT`; `mode` is `WS`, `WoS` or `M`). Runs on the `default` queue with its own `timeout_cec_classification`. CEC serves one request at a time and rebuilds its predictor per call, so send a paper's citations in one request.
- `POST /process/references/grobid`: PDF to raw and parsed references with GROBID fulltext. `include_context=true` adds, per reference, the citing sentences as `{section, citation, marker, ref_id}`; these can be posted as-is to `/classify/citations`.
- `POST /extract/references` accepts a multipart PDF with `method=grobid` (the default for uploads), plus `include_parsed` and `include_context`. Results stay index-aligned: `references` (raw strings), `parsed_references`, `contexts`. The JSON `text` body (LLM) is unchanged; mismatched combinations return 400.
- OpenCitations linking target (`target=opencitations`, also part of `all`). The connector looks up the reference's DOI through the Meta REST API, then falls back to a SPARQL title-prefix / family-name + year search on `sparql.opencitations.net/meta`. Candidate IDs are OMIDs (`br/...`). Optional `OPENCITATIONS_API` (sent as the `authorization` header; wired into `worker-linking` in compose and K8s).

### Fixed
- `GrobidClient.process_full_text` sent `consolidateReferences` / `includeRawReferences`, which GROBID ignores; it now sends `consolidateCitations` / `includeRawCitations`. Raw reference strings are now actually returned and the `consolidate_references` flag now takes effect. `process_references` still uses the old names.

### Changed
- Version bumped to 0.4.0 (package, image tags in `deployment/`). `TIMEOUT_CEC_CLASSIFICATION` is now set in `docker-compose.yml`, the ConfigMap and `.env.example`.
- `api_0.0.1.yml` (hand-written, listed endpoints that do not exist) replaced by `api_0.4.2.yml`, generated from the FastAPI app; it includes `/link/references`, `/classify/citations`, `/process/references/grobid` and the `X-API-Key` scheme.
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
