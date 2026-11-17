# Corrections to the citation-linking gold

Everything that separates the reviewed gold dataset (`yurui983/citation_linking`) from
the raw annotations (`yurui983/citation_linking_annotations`), with the reason for each
change. The raw annotations are never edited; all corrections are applied in
`build_annotation_dataset.py` (`ANNOTATION_FIXES`, `REVIEW_OVERRIDES` and
`EXCLUDED_REFERENCES`).

Sources: [issue #49](https://github.com/odoma-ch/ssh-citation-index/issues/49)
(October 2026), the cross-index review that followed the OpenCitations annotation
round, and the DOI review of the DASI/EBDA references (both November 2026).

**Rule applied throughout.** A gold record is the cited publication itself. A review
of it, another edition or version of the same text, or a record whose authors or year
do not match the reference is not a match. Where an index has no record of the cited
publication, `gold_id` is null.

| Round | What | References | Tasks changed |
|---|---|---:|---:|
| 1 | Contradictory annotation answers | 7 | 8 |
| 2 | Conflicting DOI evidence, OpenAlex vs. Matilda (issue #49) | 4 | 3 gold IDs, 3 DOIs |
| 3 | OpenAlex gold IDs that no longer resolve (issue #49) | 8 | 8 |
| 4 | Same DOI judged differently across indexes (after OpenCitations) | 13 | 11 gold IDs, 6 DOIs |
| 5 | DASI/EBDA references checked against their printed DOI | 3 | 3 gold IDs, 1 DOI |

## 1. Contradictory annotation answers

The three Argilla answers of one annotation contradicted each other, or a wrong match
had no corrected ID. One answer was corrected and the label derived again.

| Reference | Index | Change | Reason |
|---|---|---|---|
| `brill_a15e3650-8e13-4550-8b67-2158fc2df786_634` | OpenAlex | `no_match`: false → true | No candidate was returned and `no_match` was left false. Matilda and Wikidata both read "correctly not found". |
| `10.1515_zfrs-1980-0104_instance-63` (Clinard, *Comparative Crime Victimization Surveys*, 1978) | OpenAlex | `no_match`: false → true | Marked as present but no ID given. OpenAlex holds 91 Clinard works, none on victimization and none with this title: the work is absent. |
| `cex_COM-SCI_28_20` ("Magic Leap. 2018") | Matilda | `is_match_correct`: true → false | Candidate "Pixeldust Studios Reptopia Magic Leap Experience" is a different work. Matches the OpenAlex answer. |
| `cex_COM-SCI_28_20` | Wikidata | `is_match_correct`: true → false | Candidate `Q18351488` is Magic Leap the company, not a publication. |
| `10.1515_zfrs-1980-0104_instance-7` | Matilda | `is_match_correct`: true → false | Candidate is an unrelated volume on the China-Pakistan Economic Corridor. |
| `brill_7b98252c-c219-491b-8a44-3e4286c9ca47_2` ("ms lat. 8589") | OpenCitations | `is_match_correct`: true → false | Candidate `br/06020233318` is a 2011 catalogue of Hebrew manuscripts, not the cited Latin manuscript. The annotator also set `no_match`; the other three indexes read "not found". |
| `dasi_Brunner2001` | OpenAlex | `is_match_correct`: true → false | Candidate `W2750627499` is an unrelated 2017 work (Linke, "Gender - Bilder - Sanaa"). The annotator also set `no_match`; Matilda and Wikidata read "not found". |
| `10.1111_1467-6478.00057_instance-71` | Matilda | `correct_id` → `[SKIP]` | The reference text bundles three publications, like the 18 references the OpenAlex annotator skipped. The reference is excluded. |

## 2. Conflicting DOI evidence (issue #49)

Confirmed OpenAlex and Matilda matches of the same reference carried different DOIs.

| Reference | OpenAlex DOI | Matilda DOI | Decision | Reason |
|---|---|---|---|---|
| `10.1111_1467-6478.00057_instance-63` (Shanley, *Feminism, Marriage, and the Law in Victorian England*, 1989) | `10.1515/9780691215983` | `10.2307/j.ctv131btx5` | `gold_doi` = OpenAlex DOI; Matilda gold → `d6313d68-4fc7-4307-9b92-48b25363cdda` | Same book under two DOIs (De Gruyter and JSTOR). The De Gruyter record keeps the original 1989 date; the JSTOR one is dated 2020. Matilda's create date is not reliable. |
| `brill_20f2caaa-677a-42c1-8857-19e196f8c74e_331` (McKitterick, *The Carolingians and the Written Word*, 1989) | `10.1017/cbo9780511583599` | `10.1515/arbi.1991.9.1.21` | `gold_doi` = OpenAlex DOI; Matilda gold → null | The Matilda record is a review of the book by another author. The Matilda gold was cleared in November 2026; the October build had only fixed the DOI. |
| `cex_ENV-SCI_54_63` (*Competition in Local Services: Solid Waste Management*, 2000) | `10.2139/ssrn.318764` | `10.1787/649b416f-en` | No change; `gold_doi` stays null | Both are right: the SSRN entry is a posting of the OECD paper, identical apart from the cover page. With two valid DOIs none is singled out. |
| `excite_9543_10` (Bakhtin, "Toward a Methodology for the Human Sciences", 1986) | `10.7560/720466` | `10.7560/720466-008` | `gold_doi` = chapter DOI; OpenAlex gold → `W4408028697` | The cited work is a book chapter. OpenAlex had matched the whole book. |

## 3. OpenAlex gold IDs that no longer resolve (issue #49)

The annotated OpenAlex ID returned 404. Each was looked up again.

| Reference | Annotated ID | Now | Reason |
|---|---|---|---|
| `cex_PHY-AST_96_10` (Shvetsov-Shilovski et al., 2009) | `W3103705684` | `W1968542656` | Dead ID; replacement verified. |
| `excite_9543_52` (Ringmar, *Identity, Interest and Action*, 1996) | `W7038752262` | `W4233971661` | Dead ID; replacement verified. |
| `linkedbook_94` (Benardete, *Hispanic Culture and Character of the Sephardic Jews*) | `W7113369480` | `W7019729045` | Dead ID; verified book record. |
| `10.1111_1467-6478.00057_instance-38` (Contracts Review Act 1980) | `W2968428332` | null | Dead ID, no replacement found. |
| `excite_32211_33` (Schröder & Blair, 1999) | `W7078628595` | null | Dead ID, no replacement found. |
| `linkedbook_26` (Bovini, 1966) | `W6995930614` | null | Dead ID, no replacement found. |
| `linkedbook_5` (Bouwsma, *Venezia e la difesa della libertà repubblicana*) | `W7053673584` | `W618320456` | Dead ID. Replaced by the 1977 Italian translation, as cited. |
| `linkedbook_73` (Pullan, *La politica sociale della Repubblica di Venezia*, 1982) | `W7001511405` | null | Dead ID, no replacement found. |

## 4. Same DOI judged differently across indexes (November 2026)

After OpenCitations was annotated, 13 references had a candidate with the same DOI
accepted in one index and rejected in another. Each record was checked against the
index itself. Records of different works in different indexes are not conflicts and
are not listed.

| Reference | Disagreement | Decision | Reason |
|---|---|---|---|
| `excite_9543_33` (Inayatullah & Blaney, "Knowing Encounters", 1996) | Chapter DOI accepted in Matilda and OpenCitations, rejected in OpenAlex | OpenAlex gold → `W58313853` | The OpenAlex record is the chapter: both authors, pp. 65-84. |
| `10.1515_zfrs-1980-0103_instance-30` (Blankenburg, "Recht als gradualisiertes Konzept", 1980) | Accepted in OpenCitations, rejected in OpenAlex | OpenAlex gold → `W181794323` | The OpenAlex record is the chapter: 1980, pp. 83-98. |
| `cex_ENV-SCI_54_70` (Reike, Vermeulen & Witjes, 2017) | Candidate accepted in OpenAlex, rejected in Matilda and OpenCitations | OpenAlex gold `W2736219668` → `W2769483102`; `gold_doi` → `10.1016/j.resconrec.2017.08.027` | The accepted candidate was another paper (Korhonen, Honkasalo & Seppälä, *Ecological Economics*). |
| `excite_9543_34` (Katzenstein, ed., *The Culture of National Security*, 1996) | Rejected in OpenAlex and Matilda, accepted in OpenCitations | `gold_doi` → null; gold IDs unchanged | The DOI is a book review's. The OpenCitations record is the book and only carries that DOI, so its gold stands. |
| `10.1111_1467-6478.00057_instance-40` (Macpherson, *Democratic Theory*, 1973) | Rejected in OpenAlex and Matilda, accepted in OpenCitations | `gold_doi` → null; gold IDs unchanged | Same case: a book record carrying a review DOI. |
| `linkedbook_85` (Thomas-Stanford, *Early Editions of Euclid's Elements*, 1926) | Accepted in OpenCitations, rejected in OpenAlex and Matilda | OpenCitations gold `br/06701571050` → null; `gold_doi` → null | The record is a four-page 1924 article of the same title, not the 1926 book. Wikidata holds the book and keeps its gold. |
| `10.1515_zfrs-1980-0104_instance-43` (Rheinstein, *Marriage Stability, Divorce, and the Law*, 1972) | Accepted in OpenCitations, rejected in OpenAlex and Matilda | OpenCitations gold `br/06140183106` → null; Wikidata gold `Q122678044` → null; `gold_doi` → null | Both records are reviews of the book (by Drinan and by Foster). |
| `excite_9543_36` (Kratochwil, "Citizenship: On the Border of Order", 1996 chapter) | 1994 journal version accepted in OpenCitations and Wikidata, rejected in OpenAlex | OpenAlex gold → `W4299787999`; OpenCitations gold `br/062202391530` → null; Wikidata gold `Q139911205` → null | The reference cites the 1996 book chapter. OpenAlex holds the chapter record, which the annotator had not selected. OpenCitations has only the 1994 article. `gold_doi` was already the chapter's. |
| `cex_SOC-SCI_102_13` (Harvey & Rauch, "Africentric youth and family rites of passage program", 1997) | Accepted in OpenCitations and Wikidata, rejected in OpenAlex and Matilda | Reference excluded | Authors and year are Harvey & Rauch, *Health & Social Work* 22 (1997); title, journal and volume are Harvey & Hill, *Social Work* 49 (2004). The cited work cannot be told, and both papers are indexed. |
| `excite_9543_10` (Bakhtin chapter) | Whole-book DOI accepted in OpenAlex, rejected in OpenCitations | No change | Already resolved in round 2; the OpenCitations annotator had selected the chapter record. |
| `brill_7e8756fa-9241-4db4-98e1-c32ddcf154f6_1567` (Wallenfels, 1998) | Accepted in OpenAlex and Matilda, rejected in OpenCitations | No change | The OpenCitations record carries the DOI but no metadata, so it cannot be confirmed. |
| `brill_e1cc5ffe-700b-439c-bd5c-33c91446abab_670` (Rius-Camps, "Le substrat grec…") | Accepted in OpenAlex and Matilda, rejected in OpenCitations | No change | Same: an OpenCitations record without metadata. |
| `brill_869d8a52-85bf-4144-b44e-bc5686aef65c_63` (Boeckh, *Die Staatshaushaltung der Athener*, 1851) | Accepted in Matilda, rejected in OpenAlex and OpenCitations | No change | The DOI belongs to a 2023 reprint; the OpenCitations record has no metadata. |

## 5. DASI/EBDA references checked against their printed DOI (November 2026)

The 200 DASI/EBDA references are rendered from the projects' BibTeX bibliographies, and
72 of them print a DOI. That DOI was looked up in OpenAlex, Matilda and OpenCitations
for every confirmed match whose DOI differed from it, and for every task the
annotator marked as having no record. Wikidata, which lacks most of these articles,
was not checked this way. Every gold ID was also resolved again.

| Reference | Finding | Decision | Reason |
|---|---|---|---|
| `dasi_Yule2002` ("Mapping Ḥimyarite Ẓafār", 2002) | OpenAlex `W4245260361` redirects to `W3112109909` | OpenAlex gold → `W3112109909` | OpenAlex merged the annotated record into another one. |
| `dasi_Wellsted1837` ("Narrative of a Journey from the Tower of Ba-'l-haff…", 1837) | OpenCitations match carries `10.2307/1797516`, the reference `10.2307/1797510` | OpenCitations gold `br/06260917783` → null; `gold_doi` = `10.2307/1797510` | The accepted record is another article by Wellsted in the same volume (p. 102). The cited DOI is not in OpenCitations. |
| `ebda_DalleyTeissier1992` ("Tablets from the Vicinity of Emar and Elsewhere", 1992) | Marked absent in Matilda; its printed DOI finds the article there | Matilda gold → `7847d2f3-385d-497b-a716-8e030d4d3a56` | Title, year and journal match. No candidate had been shown to the annotator. The article has two DOIs (Cambridge and JSTOR); OpenAlex and OpenCitations hold it under the Cambridge one. |

The other lookups confirmed the annotations. Where a DOI resolved, it named the whole
book for a cited chapter, another chapter, or an OpenCitations record without
metadata; most found nothing.
