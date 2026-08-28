#!/usr/bin/env python3
"""Run citation_index API over PDFs in a folder.

Chain per PDF: POST /extract/text -> POST /extract/references -> POST /parse/references.
(/process/references is commented out in api.py, so the 3-stage chain is required.)

Each stage's output is written to disk as soon as it completes, under
<out>/<pdf-stem>/: text.md, references.json, parsed.json, status.json.

Usage:
    python scripts/run_pdfs.py --batch 4
    python scripts/run_pdfs.py --batch all --pdf-dir tmp --out results_api
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

# ---------------------------------------------------------------- config vars
API_BASE = os.environ.get(
    "CITATION_API",
    "https://citation-index-api-graphia-app1-staging.apps.bst2.paas.psnc.pl",
)

PDF_DIR = Path("tmp")
OUT_DIR = Path("results_api")

# stage 1: text extraction
EXTRACTOR = "pymupdf"  # pymupdf | mineru | grobid
MARKDOWN = True

# stage 2: reference extraction
EXTRACTION_METHOD = "full_text"  # full_text | semantic_sections
EXTRACTION_PROMPT = None  # None = server default; e.g. "prompts/reference_extraction.md"
EXTRACTION_TEMPERATURE = 0.3

# stage 3: reference parsing
PARSER = "llm"  # llm | grobid
PARSING_PROMPT = None  # None = server default; e.g. "prompts/reference_parsing.md"
PARSING_TEMPERATURE = 0.0

# poll deadlines = server-side task timeout (.env) + buffer
POLL_INTERVAL = 10
DEADLINE_TEXT = 300 + 120
DEADLINE_EXTRACTION = 1500 + 180
DEADLINE_PARSING = 2700 + 300

HTTP_TIMEOUT = 60

STAGES = ("1 extract text", "2 extract refs", "3 parse refs")


class StageError(RuntimeError):
    pass


def classify(resp) -> str:
    """Map a GET /jobs/{id}/status response to done | failed | pending."""
    if resp.status_code >= 500:
        return "failed"
    if resp.status_code == 202:
        return "pending"
    status = resp.json().get("status")
    if status == "completed":
        return "done"
    if status == "failed":
        return "failed"
    return "pending"


def wait(job_id: str, deadline: int) -> dict:
    """Poll status until completed, then fetch the result once."""
    end = time.time() + deadline
    while time.time() < end:
        resp = requests.get(f"{API_BASE}/jobs/{job_id}/status", timeout=HTTP_TIMEOUT)
        state = classify(resp)
        if state == "done":
            result = requests.get(f"{API_BASE}/jobs/{job_id}", timeout=HTTP_TIMEOUT)
            if result.status_code != 200:
                raise StageError(f"job {job_id} result fetch: {result.text[:300]}")
            return result.json()
        if state == "failed":
            detail = resp.text[:300]
            try:
                detail = resp.json().get("error") or detail
            except ValueError:
                pass
            raise StageError(f"job {job_id} failed: {detail}")
        time.sleep(POLL_INTERVAL)
    raise StageError(f"job {job_id} timed out after {deadline}s")


def drop_none(params: dict) -> dict:
    """Omit unset params so the API applies its own defaults."""
    return {k: v for k, v in params.items() if v is not None}


def submit(path: str, **kwargs) -> str:
    resp = requests.post(f"{API_BASE}{path}", timeout=HTTP_TIMEOUT, **kwargs)
    if resp.status_code >= 400:
        raise StageError(f"POST {path}: {resp.status_code} {resp.text[:300]}")
    return resp.json()["job_id"]


def dump(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def run_pdf(pdf: Path, out_dir: Path, bars: dict) -> dict:
    """Run the 3-stage chain for one PDF, writing each stage's output as it lands."""
    job_dir = out_dir / pdf.stem
    job_dir.mkdir(parents=True, exist_ok=True)
    out = {"pdf": pdf.name, "status": "running", "stages_done": []}

    def checkpoint(stage: str | None = None):
        if stage:
            out["stages_done"].append(stage)
            bars[stage].update(1)
        dump(job_dir / "status.json", out)

    try:
        with pdf.open("rb") as fh:
            out["text_job"] = submit(
                "/extract/text",
                files={"file": (pdf.name, fh, "application/pdf")},
                params={"extractor": EXTRACTOR, "markdown": str(MARKDOWN).lower()},
            )
        checkpoint()
        text = wait(out["text_job"], DEADLINE_TEXT)["text"]
        (job_dir / "text.md").write_text(text)
        checkpoint(STAGES[0])

        out["extraction_job"] = submit(
            "/extract/references",
            json={"text": text},
            params=drop_none(
                {
                    "method": EXTRACTION_METHOD,
                    "prompt_name": EXTRACTION_PROMPT,
                    "temperature": EXTRACTION_TEMPERATURE,
                }
            ),
        )
        checkpoint()
        ref_strings = wait(out["extraction_job"], DEADLINE_EXTRACTION)["references"]
        dump(job_dir / "references.json", ref_strings)
        out["reference_count"] = len(ref_strings)
        checkpoint(STAGES[1])

        if not ref_strings:
            out["status"] = "no_references"
            checkpoint()
            return out

        out["parsing_job"] = submit(
            "/parse/references",
            json={"references": ref_strings},
            params=drop_none(
                {
                    "parser": PARSER,
                    "prompt_name": PARSING_PROMPT,
                    "temperature": PARSING_TEMPERATURE,
                }
            ),
        )
        checkpoint()
        parsed = wait(out["parsing_job"], DEADLINE_PARSING)["references"]
        dump(job_dir / "parsed.json", parsed)
        out["parsed_count"] = len(parsed)
        out["status"] = "ok"
        checkpoint(STAGES[2])
    except (StageError, requests.RequestException, KeyError, OSError) as exc:
        out["status"] = "failed"
        out["error"] = f"{type(exc).__name__}: {exc}"
        checkpoint()
    return out


def batch_size(arg: str, total: int) -> int:
    if arg == "all":
        return max(total, 1)
    n = int(arg)
    if n < 1:
        raise ValueError("--batch must be >= 1 or 'all'")
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", default="1", help="jobs in flight at a time: number or 'all'")
    ap.add_argument("--pdf-dir", default=str(PDF_DIR))
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir)
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"no PDFs in {pdf_dir}", file=sys.stderr)
        return 1

    workers = batch_size(args.batch, len(pdfs))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(pdfs)} PDFs, {workers} at a time, API {API_BASE}")

    bars = {
        stage: tqdm(total=len(pdfs), desc=stage, position=i, unit="pdf")
        for i, stage in enumerate(STAGES)
    }
    overall = tqdm(total=len(pdfs), desc="pdfs done", position=len(STAGES), unit="pdf")

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_pdf, pdf, out_dir, bars) for pdf in pdfs]
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            overall.update(1)
            tqdm.write(
                f"[{res['status']}] {res['pdf']} "
                f"refs={res.get('reference_count', '-')} parsed={res.get('parsed_count', '-')}"
            )
            dump(out_dir / "summary.json", results)  # rewritten after every PDF

    for bar in (*bars.values(), overall):
        bar.close()

    failed = [r["pdf"] for r in results if r["status"] != "ok"]
    print(f"done: {len(results) - len(failed)} ok, {len(failed)} not ok -> {out_dir / 'summary.json'}")
    if failed:
        print("not ok: " + ", ".join(failed))
    return 1 if failed else 0


def _selfcheck():
    import tempfile
    from unittest.mock import patch

    class Bar:
        """Stand-in for tqdm; disabled tqdm bars don't count updates."""

        def __init__(self):
            self.n = 0

        def update(self, k):
            self.n += k

    class R:
        def __init__(self, code, body=None):
            self.status_code = code
            self._body = body or {}
            self.text = json.dumps(self._body)

        def json(self):
            return self._body

    assert batch_size("all", 7) == 7
    assert batch_size("3", 7) == 3
    assert classify(R(202, {"status": "started"})) == "pending"
    assert classify(R(500, {"detail": "boom"})) == "failed"
    assert classify(R(200, {"status": "completed"})) == "done"
    assert classify(R(200, {"status": "failed", "error": "x"})) == "failed"
    assert classify(R(200, {"status": "queued"})) == "pending"
    assert drop_none({"a": 1, "b": None}) == {"a": 1}

    # full chain against a fake API: stage outputs land on disk, bars advance
    results = iter(
        [{"text": "T"}, {"references": ["A", "B"]}, {"references": [{"title": "A"}]}]
    )
    with tempfile.TemporaryDirectory() as tmp:
        pdf = Path(tmp) / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        out_dir = Path(tmp) / "out"
        bars = {s: Bar() for s in STAGES}
        with patch(f"{__name__}.submit", lambda *a, **k: "job1"), patch(
            f"{__name__}.wait", lambda *a, **k: next(results)
        ):
            res = run_pdf(pdf, out_dir, bars)
        assert res["status"] == "ok", res
        assert res["stages_done"] == list(STAGES)
        job_dir = out_dir / "doc"
        assert (job_dir / "text.md").read_text() == "T"
        assert json.loads((job_dir / "references.json").read_text()) == ["A", "B"]
        assert json.loads((job_dir / "parsed.json").read_text()) == [{"title": "A"}]
        assert json.loads((job_dir / "status.json").read_text())["status"] == "ok"
        assert all(b.n == 1 for b in bars.values())

    # empty extraction -> no_references, stage 3 skipped, no parsed.json
    results = iter([{"text": "T"}, {"references": []}])
    with tempfile.TemporaryDirectory() as tmp:
        pdf = Path(tmp) / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        out_dir = Path(tmp) / "out"
        bars = {s: Bar() for s in STAGES}
        with patch(f"{__name__}.submit", lambda *a, **k: "job1"), patch(
            f"{__name__}.wait", lambda *a, **k: next(results)
        ):
            res = run_pdf(pdf, out_dir, bars)
        assert res["status"] == "no_references", res
        assert not (out_dir / "doc" / "parsed.json").exists()
    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        sys.exit(main())
