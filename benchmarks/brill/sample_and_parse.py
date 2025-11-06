# pip install neo4j pandas tqdm
from neo4j import GraphDatabase
import pandas as pd
import json, random
import os
import sys
from pathlib import Path
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from citation_index.llm.client import LLMClient
from citation_index.pipelines.reference_parsing import parse_reference_strings

# ==== CONFIG ====
URI      = "bolt://localhost:7687"         # or "neo4j+s://<host>:7687"
USER     = "neo4j"
PASSWORD = "12345678"
DB_NAME  = "brill-books-graph-neo4j5"      # <-- set your database name

N_TOTAL        = 200
MATCHED_RATIO  = 0.4225                     # ~42.25%
CONF_MIN       = 0.7                        # confidence threshold for matches

N_MATCHED   = int(round(N_TOTAL * MATCHED_RATIO))
N_UNMATCHED = N_TOTAL - N_MATCHED

# LLM CONFIG for parsing
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "https://llm.graphia-ssh.eu")
LLM_MODEL = os.getenv("LLM_MODEL", "DeepSeek-V3.1-vLLM")
LLM_API_KEY = os.getenv("LITELLM_API_KEY")

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

# ---- Matched: require ExternalPublication with confidence >= $conf_min
#      (covers both Reference->RefersTo and Reference->Cluster->RefersTo,
#       and also Publication->Cites directly to ep when no Reference exists)
QUERY_MATCHED = """
// Pick candidate EPs with sufficient confidence
MATCH (ep:ExternalPublication)
WHERE toFloat(ep.confidence) >= $conf_min

// Try to reach them via a Reference (direct or cluster)
OPTIONAL MATCH (r1:Reference)-[:RefersTo]->(ep)
OPTIONAL MATCH (r2:Reference)-[:BelongsTo]->(:Cluster)-[:RefersTo]->(ep)
WITH ep, coalesce(r1, r2) AS r

// Either we have a Reference...
OPTIONAL MATCH (p1:Publication)-[:Includes|Cites]->(r)
// ...or the Publication cites ep directly (no Reference node)
OPTIONAL MATCH (p2:Publication)-[:Cites]->(ep)
WITH ep, coalesce(p1, p2) AS p, r
WHERE p IS NOT NULL

// Collect identifiers off the ep
OPTIONAL MATCH (ep)-[:HasIdentifier]->(iid:IndustryIdentifier)
WITH p, r, ep,
     collect({
       id_type:  coalesce(iid.type, iid.types),
       id_value: coalesce(iid.id, iid.uri, iid.url, iid.name, iid.title, iid.text)
     }) AS identifiers,
     rand() AS rnd
RETURN
  coalesce(p.id, p.UUID, p.uri, p.url)      AS publication_id,
  coalesce(p.title, p.name)                 AS publication_title,
  CASE WHEN r IS NULL THEN NULL ELSE r.ref_num END                 AS ref_num,
  CASE WHEN r IS NULL THEN NULL ELSE coalesce(r.text, r.data, r.title) END AS ref_string,
  true                                      AS is_disambiguated,
  ep.title                                  AS matched_title,
  ep.year                                   AS matched_year,
  ep.publisher                              AS matched_publisher,
  ep.url                                    AS matched_url,
  ep.type                                   AS match_source,       // 'google','crossref',...
  toFloat(ep.confidence)                    AS match_confidence,
  ep.UUID                                   AS matched_uuid,
  identifiers                               AS identifiers
ORDER BY rnd
LIMIT $limit;
"""

# ---- Unmatched: references that do NOT resolve to any ep with confidence >= $conf_min
QUERY_UNMATCHED = """
MATCH (p:Publication)-[:Includes|Cites]->(r:Reference)
OPTIONAL MATCH (r)-[:RefersTo]->(ep1:ExternalPublication)
OPTIONAL MATCH (r)-[:BelongsTo]->(:Cluster)-[:RefersTo]->(ep2:ExternalPublication)
WITH p, r, [e IN [ep1, ep2] WHERE e IS NOT NULL AND toFloat(e.confidence) >= $conf_min] AS good_eps
WHERE size(good_eps) = 0
WITH p, r, rand() AS rnd
RETURN
  coalesce(p.id, p.UUID, p.uri, p.url)   AS publication_id,
  coalesce(p.title, p.name)              AS publication_title,
  r.ref_num                              AS ref_num,
  coalesce(r.text, r.data, r.title)      AS ref_string,
  false                                  AS is_disambiguated,
  NULL                                   AS matched_title,
  NULL                                   AS matched_year,
  NULL                                   AS matched_publisher,
  NULL                                   AS matched_url,
  NULL                                   AS match_source,
  NULL                                   AS match_confidence,
  NULL                                   AS matched_uuid,
  []                                     AS identifiers
ORDER BY rnd
LIMIT $limit;
"""

def run_query(query, limit):
    with driver.session(database=DB_NAME) as session:
        res = session.run(query, limit=limit, conf_min=CONF_MIN)
        return [r.data() for r in res]

matched_rows   = run_query(QUERY_MATCHED,   N_MATCHED)
unmatched_rows = run_query(QUERY_UNMATCHED, N_UNMATCHED)

# Combine & shuffle for annotation
rows = matched_rows + unmatched_rows
random.Random(42).shuffle(rows)

# Enrich: flatten identifiers and pull quick DOI/ISBN helpers
def enrich(row):
    ids = row.get("identifiers") or []
    row["identifiers_json"] = json.dumps(ids, ensure_ascii=False)

    def pick_id(kind):
        for x in ids:
            t = (x.get("id_type") or "").upper()
            if t == kind:
                return x.get("id_value")
        return None

    row["matched_doi"]  = pick_id("DOI")
    row["matched_isbn"] = pick_id("ISBN")
    return row

rows = [enrich(r) for r in rows]

df = pd.DataFrame(rows, columns=[
    "publication_id","publication_title","ref_num","ref_string",
    "is_disambiguated",
    "matched_title","matched_year","matched_publisher","matched_url",
    "match_source","match_confidence","matched_uuid",
    "matched_doi","matched_isbn","identifiers_json"
])

# Save intermediate CSV
csv_path = "brillkg_sample_200.csv"
df.to_csv(csv_path, index=False)
print(f"✅ Saved {len(df)} rows ({N_MATCHED} matched (conf≥{CONF_MIN}), {N_UNMATCHED} unmatched) to {csv_path}")

# ==== PARSING STEP ====
print("\n🔄 Starting reference parsing with LLM...")
llm_client = LLMClient(endpoint=LLM_ENDPOINT, model=LLM_MODEL, api_key=LLM_API_KEY)

# Parse references in batches for efficiency
BATCH_SIZE = 10
jsonl_records = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Parsing references"):
    ref_string = row["ref_string"]
    
    if pd.isna(ref_string) or not ref_string.strip():
        # No reference string, create empty parsed object
        parsed_data = None
    else:
        try:
            # Parse single reference
            references = parse_reference_strings(
                reference_lines=[ref_string],
                llm_client=llm_client,
                prompt_name="prompts/reference_parsing.md",
                temperature=0.0,
            )
            
            # Extract parsed reference data
            if references and references.references and len(references.references) > 0:
                ref = references.references[0]
                parsed_data = {
                    "full_title": ref.full_title or None,
                    "journal_title": ref.journal_title or None,
                    "authors": None,
                    "editors": None,
                    "publisher": ref.publisher or None,
                    "translator": None,
                    "publication_date": ref.publication_date or None,
                    "publication_place": ref.publication_place or None,
                    "volume": ref.volume or None,
                    "issue": ref.issue or None,
                    "pages": ref.pages or None,
                    "cited_range": ref.cited_range or None,
                    "footnote_number": ref.footnote_number or None,
                }
                
                # Process authors
                if ref.authors:
                    authors_list = []
                    for author in ref.authors:
                        if isinstance(author, str):
                            authors_list.append(author)
                        else:
                            authors_list.append({
                                "first_name": getattr(author, "first_name", None) or None,
                                "middle_name": getattr(author, "middle_name", None) or None,
                                "surname": getattr(author, "surname", None) or None,
                                "name_link": None,
                                "role_name": None,
                            })
                    parsed_data["authors"] = authors_list if authors_list else None
                
                # Process editors
                if ref.editors:
                    editors_list = []
                    for editor in ref.editors:
                        if isinstance(editor, str):
                            editors_list.append(editor)
                        else:
                            editors_list.append({
                                "first_name": getattr(editor, "first_name", None) or None,
                                "middle_name": getattr(editor, "middle_name", None) or None,
                                "surname": getattr(editor, "surname", None) or None,
                                "name_link": None,
                                "role_name": None,
                            })
                    parsed_data["editors"] = editors_list if editors_list else None
                
                # Process translators
                if ref.translator:
                    translators_list = []
                    for translator in ref.translator:
                        if isinstance(translator, str):
                            translators_list.append(translator)
                        else:
                            translators_list.append({
                                "first_name": getattr(translator, "first_name", None) or None,
                                "middle_name": getattr(translator, "middle_name", None) or None,
                                "surname": getattr(translator, "surname", None) or None,
                                "name_link": None,
                                "role_name": None,
                            })
                    parsed_data["translator"] = translators_list if translators_list else None
            else:
                parsed_data = None
        except Exception as e:
            tqdm.write(f"  ⚠️ Error parsing reference {idx+1}: {e}")
            parsed_data = None
    
    # Build JSONL record
    ref_id = f"brill_{row['publication_id']}_{row['ref_num']}" if not pd.isna(row['ref_num']) else f"brill_{row['publication_id']}_direct"
    
    jsonl_record = {
        "ref_id": ref_id,
        "source": "brill",
        "publication_id": row["publication_id"],
        # "publication_title": row["publication_title"],
        "ref_num": row["ref_num"] if not pd.isna(row["ref_num"]) else None,
        "original_string": ref_string if not pd.isna(ref_string) else "",
        "parsed": parsed_data,
        "linking": {
            "is_disambiguated": bool(row["is_disambiguated"]),
            "matched_title": row["matched_title"] if not pd.isna(row["matched_title"]) else None,
            "matched_year": row["matched_year"] if not pd.isna(row["matched_year"]) else None,
            "matched_publisher": row["matched_publisher"] if not pd.isna(row["matched_publisher"]) else None,
            "matched_url": row["matched_url"] if not pd.isna(row["matched_url"]) else None,
            "match_source": row["match_source"] if not pd.isna(row["match_source"]) else None,
            "match_confidence": float(row["match_confidence"]) if not pd.isna(row["match_confidence"]) else None,
            "matched_uuid": row["matched_uuid"] if not pd.isna(row["matched_uuid"]) else None,
            "matched_doi": row["matched_doi"] if not pd.isna(row["matched_doi"]) else None,
            "matched_isbn": row["matched_isbn"] if not pd.isna(row["matched_isbn"]) else None,
        }
    }
    
    jsonl_records.append(jsonl_record)

# Save as JSONL with metadata
jsonl_path = "brillkg_sample_200_parsed.jsonl"

# Count references with parsed data
refs_with_parsed = sum(1 for r in jsonl_records if r.get("parsed") is not None)

metadata = {
    "_metadata": {
        "source": "brill",
        "database": DB_NAME,
        "conf_min": CONF_MIN,
        "total_sampled": N_TOTAL,
        "matched_samples": N_MATCHED,
        "unmatched_samples": N_UNMATCHED,
        "matched_ratio": MATCHED_RATIO,
        "refs_with_parsed": refs_with_parsed,
        "created_at": pd.Timestamp.now().isoformat(),
    }
}

with open(jsonl_path, "w", encoding="utf-8") as f:
    # Write metadata as first line
    f.write(json.dumps(metadata, ensure_ascii=False) + "\n")
    # Write all records
    for record in jsonl_records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"\n✅ Saved {len(jsonl_records)} parsed references to {jsonl_path}")

driver.close()
