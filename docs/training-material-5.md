# Training Material 5

**Topic:** Automatic bibliographic data extraction with the SSH Citation Index API

**Authors:** Matteo Romanello, Giovanni Colavizza

**Classification:** Data preprocessing, data enrichment

**Short description:** This module introduces the SSH Citation Index API, an LLM-powered modular service for bibliographic data preprocessing and enrichment in Social Sciences and Humanities publications. Participants will learn to use its two core endpoints — reference extraction/parsing and citation linking against OpenCitations, Wikidata, OpenAlex, and Matilda — and understand how the pipeline handles SSH-specific challenges (e.g., multilingual sources, footnote citations).

## Learning outcomes

By the end of this module you can:

1. Authenticate with an API key and submit requests to the API, both with `curl` and with the Python client, following the asynchronous job pattern (submit, poll, fetch the result).
2. Convert a PDF to Markdown and extract the raw bibliographic references from it, choosing between the `full_text`, `semantic`, and GROBID methods.
3. Parse reference strings into structured records (authors, title, year, venue, pages, identifiers), and check that a record has the fields needed for linking.
4. Link parsed records to OpenAlex, Matilda, Wikidata, and OpenCitations, and interpret the results, including a null link.
5. Explain why linking against several databases is necessary for SSH publications, and why they need a pipeline that looks beyond a single English “References” section at the end of the PDF.

The examples below target API **v0.4.2**. The examples use the hosted instance at `https://citation-index.graphia-ssh.eu`. Interactive documentation is at [https://citation-index.graphia-ssh.eu/docs](https://citation-index.graphia-ssh.eu/docs).

## Why SSH publications need their own tooling

Commercial citation indexes are designed for the sciences, and their coverage of the humanities and social sciences is uneven. Colavizza, Peroni and Romanello ([2023](https://doi.org/10.1007/s00799-022-00327-0)) argue for an open Humanities Citation Index built around the specificities of the humanities. Among the requirements they discuss are comprehensive coverage of publication types, including books and citations to primary sources, and rich context for each citation, such as the excerpt in which it appears. They also stress multilingualism: scholarship in national languages abounds in most humanities fields, yet digital resources often fail to capture it.

The SSH Citation Index API is a step towards that goal. It extracts references from SSH publications, parses them into structured records, and links them to open scholarly databases. Its design follows from the challenges below. The method names mentioned here (`semantic`, GROBID, MinerU) are explained in Section 1.

### SSH-specific challenges

STM papers usually end with one Latin-script bibliography. SSH publications often do not. Two cases show up throughout this pipeline.

#### Multilingual sources

Bibliography headings and reference strings appear in the language of the publication: *Bibliographie*, *Literaturverzeichnis*, *Referencias*, *参考文献*, and many others. Author names, titles, and transliteration vary with the citation style and the script.

The `semantic` extraction method does not look for a single heading string. It embeds each page and compares it with example headings and example references in several languages, using the multilingual embedding model `intfloat/multilingual-e5-large-instruct`. Pages that resemble those examples, or that contain lexical cues such as a DOI, ISBN, or a four-digit year, are the ones sent to the LLM.

Parsing itself is done by the LLM (or by GROBID), so a German, French, or Italian reference string can be split into the same fields as an English one. Linking then compares normalized titles and family names, which is what makes a match possible when the local citation abbreviates a given name or punctuates a title differently from the index record.

The project evaluates this behaviour on SSH corpora that are not English-only, including EXCITE (German and English, including footnote bibliographies), CEX, and LinkedBooks.

#### Footnote citations

In many SSH genres the full reference is in a footnote or an endnote, not in a section called “References”. An in-text marker such as `(Smith et al., 2020)` is not a reference and should not be extracted; the note that contains the author, title, and year should.

Three parts of the pipeline are built for that layout:

- **Layout-aware text extraction.** MinerU and GROBID keep footnote text that a plain text dump often merges into the paragraph or drops. GROBID’s TEI records a footnote number on the bibliographic structure, and the parser copies it onto `footnote_number`.
- **Where the model is told to look.** The end-to-end extraction prompt lists footnotes, endnotes, and captions alongside the terminal bibliography, and it tells the model to ignore bare in-text citations. The default `full_text` prompt is shorter: it asks for one reference string per line and relies on the model to recognise a reference wherever it sits. For a footnote-style article, the LLM route on MinerU text is the safer choice. GROBID does find the notes, but on this kind of article it often merges two references into one string and mis-splits author names, so many of its records cannot be linked.
- **A schema that refuses empty records.** On a footnote-style PDF, guided decoding with no required fields returned references whose titles were all null. Requiring the bibliographic fields stops that silent drop. Check `count` and spot-check `full_title` and `authors` before linking; a completed job can still contain a reference that is too thin to link.

GROBID with `include_context=true` also returns the citing sentence and its section. That context is the input to `POST /classify/citations` (citation intent: method, background, or result). Intent classification is outside the two core endpoints of this module.

## How a request works

Every processing route is asynchronous. A successful `POST` does not return the bibliography. It returns a job:

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "created_at": "2026-11-03T10:00:00",
  "message": "Reference extraction job enqueued"
}
```

Poll `GET /jobs/{job_id}/status` until `status` is `completed` or `failed`. A failed job includes an `error` string. When the status is `completed`, `GET /jobs/{job_id}` returns the result. While the job is still running, that same URL responds with HTTP 202.

With `curl`, you poll by hand:

```bash
curl -s "https://citation-index.graphia-ssh.eu/jobs/$JOB_ID/status" \
  -H "X-API-Key: $CITATION_INDEX_API_KEY"

curl -s "https://citation-index.graphia-ssh.eu/jobs/$JOB_ID" \
  -H "X-API-Key: $CITATION_INDEX_API_KEY"
```

With the Python client, `wait_for_result` does the polling for you and returns the result once the job completes. It raises `JobFailedError` if the job fails and `JobTimeoutError` if it takes longer than `timeout` seconds (default 600).

```python
status = client.get_job_status(job.job_id)   # a single status check
result = client.wait_for_result(job.job_id)  # poll, then fetch the result
```

When the deployment requires an API key, send it on every request except `GET /health`. To request a key for the hosted instance, send an email to [graphia@odoma.ch](mailto:graphia@odoma.ch). This procedure may change in the future. Store your key in the environment variable `CITATION_INDEX_API_KEY` rather than pasting it into commands or notebooks:

```bash
export CITATION_INDEX_API_KEY="your-key-here"
```

Every example below is shown twice: as a `curl` command for the terminal, and as a Python snippet that uses the [SSH Citation Index API client](https://github.com/odoma-ch/ssh-citation-index-api-client). The `curl` examples pass the key as a header:

```bash
-H "X-API-Key: $CITATION_INDEX_API_KEY"
```

To use the Python client, install it and create a client once. The Python snippets below all reuse this `client` object. The client reads the key from `CITATION_INDEX_API_KEY` automatically. Pass `base_url` explicitly, because the client's default points at a staging deployment.

```bash
pip install "git+https://github.com/odoma-ch/ssh-citation-index-api-client.git"
```

The hosted API only accepts TLS 1.3, so you need a Python build linked against OpenSSL 1.1.1 or newer. Use Python 3.10 or later from python.org, Homebrew, or pyenv. The system Python 3.9 on macOS uses LibreSSL 2.8.3 and fails every call with `TLSV1_ALERT_PROTOCOL_VERSION`.

```python
from citation_index_api_client import CitationIndexClient

client = CitationIndexClient(
    "https://citation-index.graphia-ssh.eu",
    timeout=120,  # per-request HTTP timeout in seconds
)
print(client.health())
```

If you run your own instance instead, replace the base URL in the examples with your own. By default a local install listens on port **8001**, and port 8000 is reserved for MinerU, the optional PDF-to-Markdown extractor.

The combined `POST /process/references` route is not enabled. Compose the stages yourself: text, then extraction, then parsing, then linking.

## 1. Reference extraction and parsing

Extraction and parsing are two jobs. Extraction finds the bibliographic strings. Parsing turns each string into fields a linker can search on.

The choice of model behind these two jobs was informed by a benchmark. Zhu, Colavizza and Romanello ([2026](https://ceur-ws.org/Vol-4187/full-02.pdf)) compared GROBID with several recent LLMs on reference extraction, reference parsing, and end-to-end document parsing. They used three datasets that reflect SSH conditions: CEX (English journal articles), EXCITE (German and English documents, with end-of-document, footnote-only, and mixed citation layouts), and LinkedBooks (humanities references with strong stylistic and linguistic variation). Their main findings explain several choices in this lesson:

- Extraction is comparatively stable: beyond a moderate model capability, most LLMs score similarly. Parsing is harder, mainly because the model must return consistent structured output on noisy layouts. This is why parsing uses a schema with required fields.
- GROBID is typically the fastest option and does well on documents close to its training data, such as English articles with a bibliography at the end. It degrades more on other languages and on footnote or mixed layouts, where LLMs are more robust.
- The authors therefore propose routing: send well-structured documents to GROBID and escalate multilingual or footnote-heavy ones to LLMs. In this lesson you make that choice yourself, by selecting the extraction method.

### From a PDF to Markdown

`POST /extract/references` reads Markdown, unless you choose the GROBID method and upload a PDF. To obtain Markdown first:

```bash
curl -s -X POST "https://citation-index.graphia-ssh.eu/extract/text?extractor=mineru&markdown=true" \
  -H "X-API-Key: $CITATION_INDEX_API_KEY" \
  -F "file=@paper.pdf"
```

```python
job = client.extract_text("paper.pdf", extractor="mineru", markdown=True)
text = client.wait_for_result(job.job_id)["text"]
```

`extractor` is `pymupdf` (default), `mineru`, or `grobid`. MinerU and GROBID preserve more of the page layout than PyMuPDF, which matters when citations live in footnotes rather than in a terminal bibliography. Read the completed job and keep the `text` field for the next call.

### Extract the reference strings

```bash
curl -s -X POST "https://citation-index.graphia-ssh.eu/extract/references?method=full_text" \
  -H "X-API-Key: $CITATION_INDEX_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text": "# References\n\nSmith, J. (2020). Machine learning approaches. Journal of AI Research, 15(3), 245-267."}'
```

```python
markdown = (
    "# References\n\n"
    "Smith, J. (2020). Machine learning approaches. Journal of AI Research, 15(3), 245-267."
)
job = client.extract_references(markdown, method="full_text")
result = client.wait_for_result(job.job_id)
references = result["references"]
```

In a real run, pass the `text` you got from `/extract/text` instead of the toy `markdown` string above.

`method` is `full_text` (default) or `semantic`.

- `full_text` sends the document to the LLM with the reference-extraction prompt. The model returns one reference per line and nothing else.
- `semantic` first locates bibliography-like passages, then extracts from those passages. Use it when the document is long or the bibliography is not under an English heading.

A completed extraction job looks like this:

```json
{
  "references": [
    "Smith, J. (2020). Machine learning approaches. Journal of AI Research, 15(3), 245-267."
  ],
  "method": "full_text",
  "count": 1
}
```

LLM extraction is not deterministic. Two runs over the same text can return noticeably different numbers of references; on one footnote-heavy article we saw 34 in one run and 50 in another. Compare `count` with what the document suggests, and when completeness matters, run the extraction twice and look at which references differ.

To skip the LLM and let GROBID read the PDF directly, upload the file and set `method=grobid`. This route is fast and deterministic, and it is the only one that returns the citing sentences. It works best on conventionally laid-out articles with a bibliography at the end; for footnote-style articles, prefer the LLM route above. Add `include_parsed=true` when you also want structured records in the same job, and `include_context=true` when you want the citing sentence of each reference.

```bash
curl -s -X POST "https://citation-index.graphia-ssh.eu/extract/references?method=grobid&include_parsed=true" \
  -H "X-API-Key: $CITATION_INDEX_API_KEY" \
  -F "file=@paper.pdf"
```

```python
job = client.extract_references_from_pdf(
    "paper.pdf", include_parsed=True, include_context=False
)
result = client.wait_for_result(job.job_id)
references = result["references"]
parsed = result["parsed_references"]
```

`POST /process/references/grobid` is the same GROBID path with parsing always turned on.

```bash
curl -s -X POST "https://citation-index.graphia-ssh.eu/process/references/grobid?include_context=true" \
  -H "X-API-Key: $CITATION_INDEX_API_KEY" \
  -F "file=@paper.pdf"
```

```python
job = client.process_references_grobid("paper.pdf", include_context=True)
result = client.wait_for_result(job.job_id)
references, parsed, contexts = (
    result["references"], result["parsed_references"], result["contexts"]
)
```

### Parse the strings

Pass the extracted lines to `POST /parse/references`. The default parser is the LLM (`parser=llm`, temperature `0.0`). `parser=grobid` parses the same strings with GROBID instead.

```bash
curl -s -X POST "https://citation-index.graphia-ssh.eu/parse/references?parser=llm" \
  -H "X-API-Key: $CITATION_INDEX_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "references": [
      "Smith, J. (2020). Machine learning approaches. Journal of AI Research, 15(3), 245-267."
    ]
  }'
```

```python
job = client.parse_references(
    ["Smith, J. (2020). Machine learning approaches. Journal of AI Research, 15(3), 245-267."],
    parser="llm",
)
records = client.wait_for_result(job.job_id)["references"]
```

The result is a list of reference objects. Fields follow a TEI-like bibliographic record: `full_title`, `authors` (each with `first_name`, `middle_name`, `surname`), `journal_title`, `volume`, `issue`, `pages`, `publication_year`, `publisher`, `publication_place`, and `identifiers` (DOI, ISBN, ISSN, and similar). Empty fields are omitted. `publication_year` is an integer taken from the date string so later stages can filter on it.

Parsing uses a JSON schema in which the bibliographic fields are required. That constraint exists because, on footnote-heavy SSH articles, an unconstrained model has dropped every title or every author while still returning well-formed JSON. `raw` and `identifiers` stay optional.

## 2. Citation linking

Linking takes a **parsed** reference, not a raw string. `POST /link/references` searches one or more scholarly indexes and keeps a candidate only when the title, first author, and year agree closely enough with the query.

The body must include a non-empty `full_title`, at least one author, and `publication_year`. A reference that failed to parse those three fields cannot be linked; fix or drop it before this call.

### Supported publication databases

Linking is only as good as the databases it searches. The endpoint queries four of them, and you choose which ones with the `target` parameter.

| Target | Database | Size (approximate) | What the link points at |
|---|---|---|---|
| `openalex` | [OpenAlex](https://openalex.org), an open catalogue of scholarly works | 240M works (June 2025) | An OpenAlex work, plus its DOI when the record has one |
| `matilda` | Matilda, which includes CrossRef data | 155M works (*œuvres*) | A Matilda record; the DOI is read from the record’s text identifiers |
| `wikidata` | [Wikidata](https://www.wikidata.org), the open knowledge base | 38.7M scholarly articles (October 2020) | A Wikidata item; the DOI is claim P356 when present |
| `opencitations` | [OpenCitations](https://opencitations.net), open bibliographic and citation data | 91M bibliographic resources (July 2024) | An OpenCitations entity, plus its DOI when the record has one |

These sizes date from earlier counts and have probably grown since, so read them as orders of magnitude.

Why query several databases instead of one? Existing databases cover the Social Sciences and Humanities unevenly:

- Some SSH disciplines have century-long publishing traditions, so many of their publications have no persistent identifier (PID) such as a DOI to match on.
- Some publication forms, such as book chapters, are under-indexed.
- Published evidence points the same way. A study of classics ([Kokash et al., 2024](https://doi.org/10.1163/24523666-bja10036)) found about 60% of references unmatched against CrossRef and Google Books. A study of German law journals ([Boulanger et al., 2025](https://doi.org/10.2139/ssrn.5350481)) found about 60% missing from OpenAlex. In a sample of about 5,000 GoTriple documents, fewer than 60% were in OpenAlex.
- There is no existing benchmark dataset for citation linking in SSH publications.

So a null link is an ordinary result, not an error, and combining indexes raises the chance of a match. Other sources exist that this endpoint does not use: national-library aggregator APIs (BnF, GND, ICCU, British Library), OCLC WorldCat (commercial), and the Google Books API.

One reference:

```bash
curl -s -X POST "https://citation-index.graphia-ssh.eu/link/references?target=all" \
  -H "X-API-Key: $CITATION_INDEX_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "reference": {
      "full_title": "Machine learning approaches",
      "authors": [{"first_name": "J.", "surname": "Smith"}],
      "publication_year": 2020
    }
  }'
```

```python
job = client.link_references(
    {
        "full_title": "Machine learning approaches",
        "authors": [{"first_name": "J.", "surname": "Smith"}],
        "publication_year": 2020,
    },
    target="all",
)
for item in client.wait_for_links(job.job_id):
    print(item.reference["full_title"], item.links["openalex"].id)
```

Several references: send an array and set `batched=true`.

```bash
curl -s -X POST "https://citation-index.graphia-ssh.eu/link/references?batched=true&target=openalex,wikidata" \
  -H "X-API-Key: $CITATION_INDEX_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "reference": [
      {
        "full_title": "Machine learning approaches",
        "authors": [{"first_name": "J.", "surname": "Smith"}],
        "publication_year": 2020
      }
    ]
  }'
```

The Python client sets `batched=true` for you whenever you pass a list. Filter the parsed records with `is_linkable` first, because the endpoint rejects the whole batch if any record lacks a title, author, or year:

```python
from citation_index_api_client import is_linkable

job = client.link_references(
    [r for r in records if is_linkable(r)],
    target=["openalex", "wikidata"],
)
for item in client.wait_for_links(job.job_id):
    for target, link in item.links.items():
        print(target, link.id, link.doi, link.is_linked)
```

`target` accepts the names listed in the table above, the keyword `all`, or a comma-separated list. The default is `all`.

A completed job returns one entry per input reference:

```json
{
  "results": [
    {
      "reference": {
        "full_title": "Machine learning approaches",
        "authors": [{"first_name": "J.", "surname": "Smith"}],
        "publication_year": 2020
      },
      "links": {
        "openalex": {"id": "https://openalex.org/W123", "doi": "10.1000/xyz"},
        "matilda": {"id": null, "doi": null},
        "wikidata": {"id": "Q42", "doi": "10.1000/xyz"},
        "opencitations": {"id": null, "doi": null}
      }
    }
  ],
  "count": 1,
  "targets": ["openalex", "matilda", "wikidata", "opencitations"]
}
```

`id` and `doi` are `null` when that index has no accepted match. If one index errors, its entry is `{"id": null, "doi": null, "error": "..."}` and the other indexes are still returned. The job fails only when every lookup fails.

Each connector retrieves a small set of candidates. A candidate is accepted when its normalized title is similar to the query and the first author family name and publication year line up. The response keeps the single best accepted candidate per index.

## 3. Doing the same with a skill

Everything above can also be done by an AI assistant that knows the API. This works through a **skill**: a folder of instructions, and optionally helper scripts, that an assistant such as Claude loads when a task matches. Think of it as a how-to guide for one job. You describe what you want in plain language, and the assistant follows the guide to do it.

The [SSH Citation Index skill](https://github.com/odoma-ch/ssh-citation-index-skill) is such a guide for this API. It does not add new capabilities: the assistant makes the same calls you made by hand (`/extract/references`, `/parse/references`, `/link/references`) with the same `CITATION_INDEX_API_KEY`, through a small script that wraps the Python client. What it adds is the judgement the previous sections taught:

- It picks a route for your document: the LLM route on MinerU text for footnote-style or non-English publications, GROBID for conventionally laid-out articles, or only parsing and linking if you start from reference strings.
- It filters out records that cannot be linked (no title, author, or year) and reports them as skipped, instead of letting the whole batch fail.
- It treats a null link as "not found in these indexes", never as an error, and never invents an identifier.
- It warns you when two indexes return different DOIs for the same reference, for example a book and a review of it.
- It saves the results to files, including a CSV table you can open in a spreadsheet, and gives you a short summary of coverage.

### Install and use

Skills are supported by Claude Code and by Claude on the web and desktop, and by some other assistants. To install it in Claude Code, clone the repository and copy the skill folder:

```bash
git clone https://github.com/odoma-ch/ssh-citation-index-skill.git
cp -r ssh-citation-index-skill/ssh-citation-index ~/.claude/skills/
```

For Claude on the web or desktop, zip the `ssh-citation-index` folder and upload it under Settings, then Capabilities, then Skills. The skill needs the same setup as the Python client: `CITATION_INDEX_API_KEY` in your environment, and either [`uv`](https://docs.astral.sh/uv/) or Python 3.10 or later (see the TLS note in Section 1).

Then ask in plain language, for example:

> Extract the references from `paper.pdf`, parse them, and link them to OpenAlex and Wikidata. Give me a CSV.

or:

> Here are twenty reference strings. Find their DOIs.

### Check what it did

The skill is a convenience layer, not an oracle. Extraction is still done by an LLM, so check the results as you would after the manual calls: compare the number of references with what the document suggests, and spot-check a few parsed records against their raw strings. The skill's output folder holds the intermediate files (`references.json`, `parsed.json`, `links.json`), so every stage can be compared with what the `curl` or Python calls return.

## References

- Colavizza, G., Peroni, S., Romanello, M. (2023). The case for the Humanities Citation Index (HuCI): a citation index by the humanities, for the humanities. *International Journal on Digital Libraries* 24, 191–204. [https://doi.org/10.1007/s00799-022-00327-0](https://doi.org/10.1007/s00799-022-00327-0)
- Boulanger, C., Fejzo, D., Rimmert, C. (2025). *Law Doesn’t Count? Measuring Bibliometric Coverage of German Law Journals.* [https://doi.org/10.2139/ssrn.5350481](https://doi.org/10.2139/ssrn.5350481)
- Kokash, N., Romanello, M., Suyver, E., Colavizza, G. (2024). “The Brill Knowledge Graph: A Database of Bibliographic References and Index Terms Extracted from Books in Humanities and Social Sciences”. *Research Data Journal for the Humanities and Social Sciences.* [https://doi.org/10.1163/24523666-bja10036](https://doi.org/10.1163/24523666-bja10036)
- Zhu, Y., Colavizza, G., Romanello, M. (2026). *Benchmarking Large Language Models on Reference Extraction and Parsing in the Social Sciences and Humanities.* In: SCOLIA ’26, Second International Workshop on Scholarly Information Access, CEUR Workshop Proceedings, vol. 4187. [https://ceur-ws.org/Vol-4187/full-02.pdf](https://ceur-ws.org/Vol-4187/full-02.pdf)

## Further reading in this repository

- [README](../README.md) — install the API, the worker queues, and the current endpoint table.
- [ARCHITECTURE](../ARCHITECTURE.md) — how a job moves from FastAPI through Redis to the extraction, LLM, and linking workers.
- `src/citation_index/api.py` — request bodies and query parameters.
- `prompts/reference_extraction.md` and `prompts/reference_parsing.md` — the default LLM instructions.
- `benchmarks/README.md` — the SSH evaluation corpora.
