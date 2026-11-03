# Training Material 5

**Topic:** Automatic bibliographic data extraction with the SSH Citation Index API

**Authors:** Matteo Romanello, Giovanni Colavizza

**Classification:** Data preprocessing, data enrichment

**Short description:** This module introduces the SSH Citation Index API, an LLM-powered modular service for bibliographic data preprocessing and enrichment in Social Sciences and Humanities publications. Participants will learn to use its two core endpoints — reference extraction/parsing and citation linking against OpenCitations, Wikidata, OpenAlex, and Matilda — and understand how the pipeline handles SSH-specific challenges (e.g., multilingual sources, footnote citations).

## Learning outcomes

By the end of this module you can:

1. Submit a publication’s text to the API and retrieve a list of raw bibliographic references.
2. Parse those strings into structured reference records (authors, title, year, venue, pages, identifiers).
3. Link a parsed record to OpenAlex, Matilda, Wikidata, and OpenCitations.
4. Explain why SSH publications need a pipeline that looks beyond a single English “References” section at the end of the PDF.

The examples below target API **v0.4.2**. Interactive documentation for a running instance is at `/docs` (locally, [http://localhost:8001/docs](http://localhost:8001/docs)).

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

When the deployment sets `API_KEY`, send it on every request except `GET /health`:

```bash
-H "X-API-Key: $API_KEY"
```

Locally the API listens on port **8001**. Port 8000 is reserved for MinerU, the optional PDF-to-Markdown extractor.

The combined `POST /process/references` route is not enabled. Compose the stages yourself: text, then extraction, then parsing, then linking.

## 1. Reference extraction and parsing

Extraction and parsing are two jobs. Extraction finds the bibliographic strings. Parsing turns each string into fields a linker can search on.

### From a PDF to Markdown

`POST /extract/references` reads Markdown, unless you choose the GROBID method and upload a PDF. To obtain Markdown first:

```bash
curl -s -X POST "http://localhost:8001/extract/text?extractor=mineru&markdown=true" \
  -H "X-API-Key: $API_KEY" \
  -F "file=@paper.pdf"
```

`extractor` is `pymupdf` (default), `mineru`, or `grobid`. MinerU and GROBID preserve more of the page layout than PyMuPDF, which matters when citations live in footnotes rather than in a terminal bibliography. Read the completed job and keep the `text` field for the next call.

### Extract the reference strings

```bash
curl -s -X POST "http://localhost:8001/extract/references?method=full_text" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text": "# References\n\nSmith, J. (2020). Machine learning approaches. Journal of AI Research, 15(3), 245-267."}'
```

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

To skip the LLM and let GROBID read the PDF directly, upload the file and set `method=grobid`. Add `include_parsed=true` when you also want structured records in the same job, and `include_context=true` when you want the citing sentence of each reference.

```bash
curl -s -X POST "http://localhost:8001/extract/references?method=grobid&include_parsed=true" \
  -H "X-API-Key: $API_KEY" \
  -F "file=@paper.pdf"
```

`POST /process/references/grobid` is the same GROBID path with parsing always turned on.

### Parse the strings

Pass the extracted lines to `POST /parse/references`. The default parser is the LLM (`parser=llm`, temperature `0.0`). `parser=grobid` parses the same strings with GROBID instead.

```bash
curl -s -X POST "http://localhost:8001/parse/references?parser=llm" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "references": [
      "Smith, J. (2020). Machine learning approaches. Journal of AI Research, 15(3), 245-267."
    ]
  }'
```

The result is a list of reference objects. Fields follow a TEI-like bibliographic record: `full_title`, `authors` (each with `first_name`, `middle_name`, `surname`), `journal_title`, `volume`, `issue`, `pages`, `publication_year`, `publisher`, `publication_place`, and `identifiers` (DOI, ISBN, ISSN, and similar). Empty fields are omitted. `publication_year` is an integer taken from the date string so later stages can filter on it.

Parsing uses a JSON schema in which the bibliographic fields are required. That constraint exists because, on footnote-heavy SSH articles, an unconstrained model has dropped every title or every author while still returning well-formed JSON. `raw` and `identifiers` stay optional.

## 2. Citation linking

Linking takes a **parsed** reference, not a raw string. `POST /link/references` searches one or more scholarly indexes and keeps a candidate only when the title, first author, and year agree closely enough with the query.

The body must include a non-empty `full_title`, at least one author, and `publication_year`. A reference that failed to parse those three fields cannot be linked; fix or drop it before this call.

One reference:

```bash
curl -s -X POST "http://localhost:8001/link/references?target=all" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "reference": {
      "full_title": "Machine learning approaches",
      "authors": [{"first_name": "J.", "surname": "Smith"}],
      "publication_year": 2020
    }
  }'
```

Several references: send an array and set `batched=true`.

```bash
curl -s -X POST "http://localhost:8001/link/references?batched=true&target=openalex,wikidata" \
  -H "X-API-Key: $API_KEY" \
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

`target` accepts `openalex`, `matilda`, `wikidata`, `opencitations`, the keyword `all`, or a comma-separated list. The default is `all`.

| Target | What the link points at |
|---|---|
| `openalex` | An OpenAlex work, plus its DOI when the record has one |
| `matilda` | A Matilda record; the DOI is read from the record’s text identifiers |
| `wikidata` | A Wikidata item; the DOI is claim P356 when present |
| `opencitations` | An OpenCitations entity, plus its DOI when the record has one |

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

## SSH-specific challenges

STM papers usually end with one Latin-script bibliography. SSH publications often do not. Two cases show up throughout this pipeline.

### Multilingual sources

Bibliography headings and reference strings appear in the language of the publication: *Bibliographie*, *Literaturverzeichnis*, *Referencias*, *参考文献*, and many others. Author names, titles, and transliteration vary with the citation style and the script.

The `semantic` extraction method does not look for a single heading string. It embeds each page and compares it with example headings and example references in several languages, using the multilingual embedding model `intfloat/multilingual-e5-large-instruct`. Pages that resemble those examples, or that contain lexical cues such as a DOI, ISBN, or a four-digit year, are the ones sent to the LLM.

Parsing itself is done by the LLM (or by GROBID), so a German, French, or Italian reference string can be split into the same fields as an English one. Linking then compares normalized titles and family names, which is what makes a match possible when the local citation abbreviates a given name or punctuates a title differently from the index record.

The project evaluates this behaviour on SSH corpora that are not English-only, including EXCITE (German and English, including footnote bibliographies), CEX, and LinkedBooks.

### Footnote citations

In many SSH genres the full reference is in a footnote or an endnote, not in a section called “References”. An in-text marker such as `(Smith et al., 2020)` is not a reference and should not be extracted; the note that contains the author, title, and year should.

Three parts of the pipeline are built for that layout:

- **Layout-aware text extraction.** MinerU and GROBID keep footnote text that a plain text dump often merges into the paragraph or drops. GROBID’s TEI records a footnote number on the bibliographic structure, and the parser copies it onto `footnote_number`.
- **Where the model is told to look.** The end-to-end extraction prompt lists footnotes, endnotes, and captions alongside the terminal bibliography, and it tells the model to ignore bare in-text citations. The default `full_text` prompt is shorter: it asks for one reference string per line and relies on the model to recognise a reference wherever it sits. For a long footnote-style article, `method=semantic` or `method=grobid` is the more reliable choice.
- **A schema that refuses empty records.** On a footnote-style PDF, guided decoding with no required fields returned references whose titles were all null. Requiring the bibliographic fields stops that silent drop. Check `count` and spot-check `full_title` and `authors` before linking; a completed job can still contain a reference that is too thin to link.

GROBID with `include_context=true` also returns the citing sentence and its section. That context is the input to `POST /classify/citations` (citation intent: method, background, or result). Intent classification is outside the two core endpoints of this module.

## Suggested exercise

Run one SSH article that has a footnote bibliography, and one that has a conventional reference list, through the same four calls.

1. `POST /extract/text` with MinerU, and keep the Markdown.
2. `POST /extract/references` twice on that Markdown: once with `method=full_text`, once with `method=semantic`. Compare the two lists. Then run `method=grobid` on the original PDF and compare again.
3. `POST /parse/references` on the list you trust. Confirm that `full_title`, the first author’s surname, and `publication_year` are present.
4. `POST /link/references?target=all` on two or three of those records. Note which indexes return an `id`, which return only a DOI, and which return null. A null link is an ordinary outcome for a local-language monograph that none of the four indexes hold.

The executable walkthrough in `examples/citation_index_api_guide.ipynb` runs this sequence against a live stack and fails if a stage returns an invalid result.

## Further reading in this repository

- [README](../README.md) — install the API, the worker queues, and the current endpoint table.
- [ARCHITECTURE](../ARCHITECTURE.md) — how a job moves from FastAPI through Redis to the extraction, LLM, and linking workers.
- `src/citation_index/api.py` — request bodies and query parameters.
- `prompts/reference_extraction.md` and `prompts/reference_parsing.md` — the default LLM instructions.
- `benchmarks/README.md` — the SSH evaluation corpora.
