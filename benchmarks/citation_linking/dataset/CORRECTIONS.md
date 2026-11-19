# Corrections to the citation-linking gold

Everything that separates the reviewed gold dataset (`yurui983/citation_linking`) from
the raw annotations (`yurui983/citation_linking_annotations`), with the reason for each
change. The raw annotations are never edited; all corrections live in `corrections.py`
(`ANNOTATION_FIXES`, `EXCLUDED_REFERENCES`, `EXCLUDED_TASKS`, `REVIEW_OVERRIDES`) and are
applied by `build_annotation_dataset.py`.

Sources: [issue #49](https://github.com/odoma-ch/ssh-citation-index/issues/49)
(October 2026), the cross-index review that followed the OpenCitations annotation
round, the DOI review of the DASI/EBDA references, and the check of every label
against the frozen candidate pool (all November 2026).

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
| 6 | Labels checked against the frozen candidate pool | 68 | 73 gold IDs, 9 DOIs, 3 tasks and 1 reference excluded |

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
| `excite_9543_34` (Katzenstein, ed., *The Culture of National Security*, 1996) | Rejected in OpenAlex and Matilda, accepted in OpenCitations | `gold_doi` → null; gold IDs unchanged | The DOI is a book review's. The OpenCitations record is the book and only carries that DOI, so its gold stands. Revised in round 6 (6f). |
| `10.1111_1467-6478.00057_instance-40` (Macpherson, *Democratic Theory*, 1973) | Rejected in OpenAlex and Matilda, accepted in OpenCitations | `gold_doi` → null; gold IDs unchanged | Same case: a book record carrying a review DOI. |
| `linkedbook_85` (Thomas-Stanford, *Early Editions of Euclid's Elements*, 1926) | Accepted in OpenCitations, rejected in OpenAlex and Matilda | OpenCitations gold `br/06701571050` → null; `gold_doi` → null | The record is a four-page 1924 article of the same title, not the 1926 book. Wikidata holds the book and keeps its gold. |
| `10.1515_zfrs-1980-0104_instance-43` (Rheinstein, *Marriage Stability, Divorce, and the Law*, 1972) | Accepted in OpenCitations, rejected in OpenAlex and Matilda | OpenCitations gold `br/06140183106` → null; Wikidata gold `Q122678044` → null; `gold_doi` → null | Both records are reviews of the book (by Drinan and by Foster). |
| `excite_9543_36` (Kratochwil, "Citizenship: On the Border of Order", 1996 chapter) | 1994 journal version accepted in OpenCitations and Wikidata, rejected in OpenAlex | OpenAlex gold → `W4299787999`; OpenCitations gold `br/062202391530` → null; Wikidata gold `Q139911205` → null | The reference cites the 1996 book chapter. OpenAlex holds the chapter record, which the annotator had not selected. OpenCitations has only the 1994 article. `gold_doi` was already the chapter's. |
| `cex_SOC-SCI_102_13` (Harvey & Rauch, "Africentric youth and family rites of passage program", 1997) | Accepted in OpenCitations and Wikidata, rejected in OpenAlex and Matilda | Reference excluded | Authors and year are Harvey & Rauch, *Health & Social Work* 22 (1997); title, journal and volume are Harvey & Hill, *Social Work* 49 (2004). The cited work cannot be told, and both papers are indexed. |
| `excite_9543_10` (Bakhtin chapter) | Whole-book DOI accepted in OpenAlex, rejected in OpenCitations | No change | Already resolved in round 2; the OpenCitations annotator had selected the chapter record. |
| `brill_7e8756fa-9241-4db4-98e1-c32ddcf154f6_1567` (Wallenfels, 1998) | Accepted in OpenAlex and Matilda, rejected in OpenCitations | No change | The OpenCitations record carries the DOI but no metadata, so it cannot be confirmed. |
| `brill_e1cc5ffe-700b-439c-bd5c-33c91446abab_670` (Rius-Camps, "Le substrat grec…") | Accepted in OpenAlex and Matilda, rejected in OpenCitations | No change | Same: an OpenCitations record without metadata. |
| `brill_869d8a52-85bf-4144-b44e-bc5686aef65c_63` (Boeckh, *Die Staatshaushaltung der Athener*, 1851) | Accepted in Matilda, rejected in OpenAlex and OpenCitations | No change | The DOI belongs to a 2023 reprint; the OpenCitations record has no metadata. Revised in round 6 (6e). |

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

## 6. Labels checked against the frozen candidate pool (November 2026)

The GLiNER decider learns from the frozen candidate pool, so every label must be right
for that pool. The pool was searched after annotation (2026-11), and earlier rounds had
only reconciled DOIs, not checked each index's accepted record. Each task below was
checked by hand against its pool candidates, with these rules:

- A review of the cited book, or another work, is never the gold.
- A reprint or reissue is the gold when all metadata but the date agree; a record with
  the same DOI is the gold even if the pages differ.
- A translation is not the original's record.
- A Wikidata work item is a valid gold when nothing in the citation or the item
  conflicts; where the pool also holds the cited edition, that edition is the gold.
- A label must follow from the citation and the candidate texts alone. A short title or
  a missing subtitle is the same title; a missing field (year, container) is missing,
  not a mismatch; an omitted editor or reviser is acceptable. Where deciding needs
  outside knowledge or several candidates are equally plausible, the task is excluded.
- Errors and noise in the reference (footnote lead-ins such as "19 Beispiel von") are
  not corrected.

### 6a. `none` labels whose record is in the pool (52 tasks)

Annotators marked these as absent from the index, but the pool holds the cited record:
it entered the index after annotation (OpenAlex's 2025 rebuild, Matilda's bulk imports
of 2026-07 and 2026-10), the annotator's search missed it, or it differs only in date.

| Reference | Index | Was | Now | Reason |
|---|---|---|---|---|
| `10.1515_zfrs-1980-0104_instance-15` (15 Näher dazu Zelditch, Intelligible Comparisons, in: Vallier 267 (270 ff.).) | matilda | none | `9e0e05af-eda6-4c45-86de-8a9caeedeec3` | Record was in the index; the annotator's search missed it. |
| `brill_0172ea14-22fd-4164-a417-cb923c77acb3_588` (Marshall, Peter. Beliefs and the Dead in Reformation England (Oxford, 2005).) | matilda | none | `d53289af-4344-4464-b144-6216065c7b30` | Record added to the index after annotation; only the date differs. |
| `brill_5b7ade43-7b83-4597-a458-042d8a99632f_64` (——, ‘ Why are institutions the “Carriers of history”?: path Dependence and the Evolution  ) | openalex | none | `W1970677847` | Record was in the index; the annotator's search missed it. |
| `brill_a63e6f1f-cf2e-44e5-bab5-0709ecf2fc82_22` (vonAlbrecht, M. 1994. Geschichte der römischen Literatur. Vol. 1. 2nd ed. Munich.) | matilda | none | `da689b31-3d6c-4317-b1e3-7546701098b1` | Another Matilda record of the same book (the annotated one is not in the pool). |
| `cex_ENV-SCI_54_70` (D Reike, W J Vermeulen, S Witjes. The circular economy: New or Refurbished as CE 3.0?. Res) | wikidata | none | `Q125841968` | Same work; only the date differs. |
| `cex_PHY-AST_96_14` (K Y Huang, Q Z Xia, L B Fu. Survival window for atomic tunneling ionization with elliptica) | matilda | none | `2e526547-5a7d-4142-b9ed-2b7526059013` | Record added to the index after annotation. |
| `excite_9543_0` (Adler, Emanuel (1997). Seizing the Middle Ground: Constructivism in International Relation) | matilda | none | `c50d9fcd-de90-45c3-8a06-f8ac54c06a4c` | Record was in the index; the annotator's search missed it. |
| `dasi_Edens1999` (Edens, Christopher. 1999. “The Bronze Age of Highland Yemen: Chronological and Spatial Var) | matilda | none | `496d5fc1-4c2b-4e40-85fe-e5e336fbc02c` | Record added to the index after annotation. |
| `dasi_Edens1999` (Edens, Christopher. 1999. “The Bronze Age of Highland Yemen: Chronological and Spatial Var) | wikidata | none | `Q57476809` | Record was in the index; the annotator's search missed it. |
| `10.1111_1467-6478.00080_instance-48` (54 P. Harris, ‘Curriculum Development in Legal Studies’ (1986) 20 The Law Teacher 110 at 1) | matilda | none | `05a57523-0145-4be5-9d53-d7424530def4` | Record added to the index after annotation. |
| `10.1515_zfrs-1980-0104_instance-66` (66 Siehe Podgdrecki, Legal Consciousness as a Research Problem, European Yearbook in Law a) | matilda | none | `f467924c-f203-46c6-bbb1-bc6082de0817` | Record added to the index after annotation. |
| `cex_DEN_35_15` (S S Chukkapalli, I M Velsko, M F Rivera-Kweh, H Larjava, A R Lucas, L Kesavalu. Global TLR) | openalex | none | `W2397755785` | Same work; only the date differs. |
| `cex_ENV-SCI_54_65` (L B Pollans. Trapped in trash: “modes of governing”and barriers to transitioning to sustai) | openalex | none | `W2761541636` | Record was in the index; the annotator's search missed it. |
| `excite_9543_14` (Bell, Michael & Gardiner, Michael (Eds.) (1998). Bakhtin and the Human Sciences. London: S) | matilda | none | `607b19cd-71e0-4374-8020-a1700ed6579d` | Record added to the index after annotation. |
| `brill_4c6ba321-d757-4ab1-b4e9-568fd2087e41_285` (Mitchell, Margaret M. “Gentile Christianity.” Pages 103–124 in Cambridge History of Christ) | openalex | none | `W4210995859` | Same work; the record lacks the container, which is missing, not a mismatch. |
| `brill_969dc0be-9925-4ee3-a7ec-c98285b071f6_484` (Mullen,  Alex  and  Patrick  James,  eds.  Multilingualism  in  the  Graeco-Roman  Worlds.) | matilda | none | `44ed92bb-6417-4921-9b40-028e29529880` | Record was in the index; the annotator's search missed it. |
| `cex_ENV-SCI_54_31` (N Gregson, M Crang, S Fuller, H Holmes. Interrogating the circular economy: The moral econ) | wikidata | none | `Q57551678` | Record was in the index; the annotator's search missed it. |
| `excite_32211_14` ("• Roth, Roland (2003): Die dunklen Seiten der Zivilgesellschaft. Grenzen einer zivilgesel) | matilda | none | `99d5aa2d-3fb2-4a47-9ab2-1ec0e6090d3c` | Record added to the index after annotation. |
| `excite_32211_14` ("• Roth, Roland (2003): Die dunklen Seiten der Zivilgesellschaft. Grenzen einer zivilgesel) | openalex | none | `W2565235294` | Pages differ, same DOI. |
| `excite_9543_23` (Emirbayer, Mustafa (1997). Manifesto for a relational sociology. American Journal of Socio) | wikidata | none | `Q139686849` | Record added to the index after annotation. |
| `ebda_Matthiae1969` (Matthiae, P. (1969). ‘Empreintes d'un cylindre paléosyrien de tell Mardikh’. Syria 46, pp.) | matilda | none | `8fb0ce6b-96bb-42dc-aaee-297528f6f4aa` | Record added to the index after annotation. |
| `10.1111_1467-6478.00080_instance-18` (20 F.R. Leavis, Education and the University (1948) 28. Leavis’s view was narrowly nationa) | openalex | none | `W2970212067` | Same title, first author and year. |
| `10.1515_zfrs-1980-0104_instance-33` (33 Blankenburg, Task Contingencies and National Administrative Culture as Determinants for) | openalex | none | `W7006124636` | Same title, first author and year. |
| `10.1515_zfrs-1980-0104_instance-43` (43 Rheinstein, Marriage Stability, Divorce, and the Law (Chicago 1972).) | openalex | none | `W6983264735` | Same title, first author and year. |
| `brill_275e72b8-c496-4d90-959e-40657ca53810_134` (Smyth,  H.W.,  Greek Grammar (Cambridge,  Mass.  1920:  reprinted  1971)) | openalex | none | `W7204422925` | Exact title "Greek grammar" and author; the record has no year, which is missing, not a mismatch. Not `W7028603349` ("A Greek grammar for colleges"). |
| `brill_624697a1-c86b-40fb-91d0-8a89d95dc958_138` (Osborne,  C.,  Rethinking Early Greek Philosophy: Hippolytus ef Rome and the Presocratics,) | openalex | none | `W7070867014` | Same title, first author and year. |
| `brill_81977646-2ce4-4c77-8ee6-0ef78a9cd358_588` (Momigliano,  A. 1975. Alien Wisdom: The Limits of Hellenization. Cambridge and London: Cam) | openalex | none | `W7073812697` | Same title, first author and year. |
| `cex_COM-SCI_28_29` (Justus Thies, Michael Zollhofer, Marc Stamminger, Christian Theobalt, Matthias Niessner. F) | openalex | none | `W7028184114` | Same title, first author and year. |
| `excite_9543_52` (Ringmar, Erik (1996). Identity, interest and action. A cultural explanation of Sweden's in) | matilda | none | `faaaee23-6c67-46f3-a4f8-01efb0d11aeb` | Same title, first author and year. |
| `linkedbook_99` (Horwitz, Henry. Parliament, Policy and Politics in the Reign of William III. Manches - ter) | openalex | none | `W7033185218` | Same title, first author and year. |
| `cex_ENV-SCI_54_29` (S Graham, S Marvin. Splintering urbanism. 2001) | opencitations | none | `br/06019323514` | Another OpenCitations record of the same book; the annotated `br/06703466811` is not in the pool. |
| `dasi_Robin2001` (Robin, C. J. (2001). ‘« La caravane yéménite et syrienne » dans une inscription de l’Arabi) | matilda | none | `463f02a8-d09b-4992-b784-3a9782196e22` | Same title, first author and year. |
| `ebda_Barstad1978` (Barstad, H. M. (1978). ‘Utgravningene i Tell Mardikh/Ebla i Nordsyria. En kort orientering) | openalex | none | `W1541528659` | Same title, first author and year. |
| `ebda_Biga2014d` (Biga, M. G. (2014). ‘Inherited Space - Third Millennium Political and Cultural Landscape’.) | openalex | none | `W2492327343` | Same title, first author and year. |
| `ebda_Biga2014d` (Biga, M. G. (2014). ‘Inherited Space - Third Millennium Political and Cultural Landscape’.) | wikidata | none | `Q127150057` | Same title, first author and year. |
| `ebda_Dahood1982a` (Dahood, M. (1982). ‘Philological Observations on Five Biblical Texts’. Biblica 63, pp. 390) | openalex | none | `W7070538371` | Same title, first author and year. |
| `ebda_Dolce2025` (Dolce, R. (2025). ‘Ancient Images in Mesopotamia: Ancient Gaze and Present Observation Bet) | matilda | none | `015dde3d-980e-41fb-851e-abd098e9b7a1` | Same title, first author and year. |
| `ebda_Dolce2025` (Dolce, R. (2025). ‘Ancient Images in Mesopotamia: Ancient Gaze and Present Observation Bet) | openalex | none | `W7140359981` | Same title, first author and year. |
| `ebda_Matthiae1982` (Matthiae, Paolo. 1982. “Die Fürstengräber des Palastes Q in Ebla.” Antike Welt 13 (1): 2–1) | openalex | none | `W7046354894` | Same title, first author and year. |
| `ebda_Matthiae2019d` (Matthiae, Paolo. 2019. “A Problem of Iconology: A Note on the Banquets of the Old Syrian B) | matilda | none | `88728b69-4e79-4304-b639-08bd9dd2c6db` | Same title, first author and year. |
| `ebda_Micale2018` (Micale, Maria Gabriella. 2018. “Clay Figurines in the Persian Achaemenid Near East as Seen) | matilda | none | `08db4245-1356-4f01-86cc-5d1e0286169f` | Same title, first author and year. |
| `10.1111_1467-6478.00057_instance-41` (43 For example, N. Howell, ‘“Sexually Transmitted Debt”: A Feminist Analysis of Laws Regul) | openalex | none | `W7062005228` | Same title, first author and year. |
| `excite_18417_10` (Gebhard, Ulrich (1994), Kind und Natur: die Bedeutung der Natur für die psychische Entwick) | matilda | none | `05ecb5f1-d2e8-4976-912c-842eacf5f876` | Same title, first author and year. |
| `excite_18417_10` (Gebhard, Ulrich (1994), Kind und Natur: die Bedeutung der Natur für die psychische Entwick) | openalex | none | `W4239222949` | Same title, first author and year. |
| `linkedbook_105` (Jones, Clyve, ed. A Pillar of the Constitution: The House of Lords in British Politics 164) | openalex | none | `W619234676` | Same title, first author and year. |
| `ebda_AscalonePeyronel2000` (Ascalone, Enrico and Luca Peyronel. 2000. “The Eblaite Metrology in the Middle Bronze Age.) | openalex | none | `W7037298554` | Same title, first author and year. |
| `ebda_Fales1984a` (Fales, F. M. (1984). ‘La radice *mzl nei testi di Ebla’. Studi Epigrafici e Linguistici 1,) | openalex | none | `W7029531406` | Same title, first author and year. |
| `ebda_Pinnock1979` (Pinnock, Frances. 1979. “Nota sui ‘sonagli’ della ‘Tomba del Signore dei capridi’.” Studi ) | openalex | none | `W7030440895` | Same title, first author and year. |
| `10.1515_zfrs-1980-0104_instance-19` (19 Beispiel von Carbonnier, Sociologie juridique (Paris 1972) 188 N. 1.) | openalex | none | `W7071678455` | "19 Beispiel von" is footnote noise; author Carbonnier, title and year match. |
| `excite_18417_31` (Preuss, Sigrun (1991), Umweltkatastrophe Mensch, Asanger/Heidelberg.) | openalex | none | `W7016054603` | The record only adds a subtitle; same author and year. |
| `dasi_Stein2003a` (Stein, P. (2003). Untersuchungen zur Phonologie und Morphologie des Sabäischen. Series: Ep) | openalex | none | `W7062438958` | Same title, first author and year. |
| `ebda_Matthiae2019d` (Matthiae, Paolo. 2019. “A Problem of Iconology: A Note on the Banquets of the Old Syrian B) | openalex | none | `W4402107120` | Same title, first author and year. |

### 6b. Accepted records that are not the cited work (15 tasks)

Mostly reviews (Wikidata items most often, then *Choice* and JSTOR reviews). Where the
pool holds the book itself, it is now the gold.

| Reference | Index | Was | Now | Reason |
|---|---|---|---|---|
| `brill_591b2204-041d-4d3f-937f-8ef19f7f73bf_214` (Euripides. Suppliant Women. Electra. Heracles. Edited and translated by David Kovacs. LCL ) | openalex | `W2802550434`, a *Classical World* review | `W431884289` | The book is in the pool: title, Euripides and Kovacs, 1998, book. |
| `cex_SOC-SCI_102_20` (E Martin, J Martin. Spirituality and the black helping tradition. 2002) | openalex | `W415486378`, a *Choice* review | none | The book is not in the pool. |
| `10.1111_1467-6478.00057_instance-39` (41 J. Nedelsky, Private Property and the Limits of American Constitutionalism: The Madison) | openalex | `W2323906362`, a *Choice* review | none | All ten candidates are reviews. |
| `10.1111_1467-6478.00057_instance-62` (66 For detailed discussion of the ramifications, see L.J. Weitzman, The Divorce Revolution) | wikidata | `Q120837750`, a *BJS* review | none | All candidates are reviews or other works. |
| `brill_0172ea14-22fd-4164-a417-cb923c77acb3_588` (Marshall, Peter. Beliefs and the Dead in Reformation England (Oxford, 2005).) | wikidata | `Q58538256`, a *History: Reviews of New Books* review | none | Only candidate. |
| `brill_2841ce8b-9934-4665-8c6c-fc3fca5b31a4_48` (Deubner,  L.  1932.  Attische  Feste,  Berlin.) | wikidata | `Q60200406`, a 1933 *JHS* review | none | Only candidate. |
| `cex_ECO-ECO-FIN_41_14` (Jeff Ely, Alexander Frankel, Emir Kamenica. Suspense and Surprise. Journal of Political Ec) | wikidata | `Q48090924`, Verleger et al. 1994 | none | A different article. |
| `excite_9543_52` (Ringmar, Erik (1996). Identity, interest and action. A cultural explanation of Sweden's in) | opencitations | `br/06019625568` (book metadata, DOI of a *Contemporary Sociology* review) | `br/062403046395` | `br/062403046395` is the book under its own DOI. |
| `excite_9543_20` (Carr, Edward H. (1939/1964). The Twenty Years Crisis 1919-1939. An Introduction to the Stu) | wikidata | `Q127204316`, a 2019 *SAIS Review* article on the book | none | Only candidate. |
| `brill_c90e527d-b3cc-4ae0-b2c5-f959ba70584e_106` (Frawley, William. Linguistic Semantics. Hillsdale, NJ: L. Erlbaum Associates, 1992.) | openalex | `W1997834200`, a *Journal of Linguistics* review | `W4236503687` | `W4236503687` is the book (2013 reissue; only the date differs). |
| `brill_63aad500-2bf4-4cb4-ae2b-14a84d24bef3_1126` (Vogt-Spira, G. 1992. Dramaturgie des Zufalls: Tyche und Handeln in der Komödie Menanders () | wikidata | `Q60201788`, a 1995 review | none | Only candidate. |
| `brill_f288353a-0a2e-4095-a902-9115298934f4_298` (Radt, W. (1988): Pergamon: Geschichte und Bauten, Funde und Erforschung einer antiken Metr) | wikidata | `Q60201663`, a 1990 review | none | Only candidate. |
| `excite_9543_58` (Todorov, Tzvetan (1981). Mikhaïl Bakhtine. Le principe dialogique suivi de Écrits du Cercl) | wikidata | `Q139895273`, a 1983 review | none | Only candidate. |
| `excite_9543_62` (Weber, Cynthia (2001). International Relations Theory. A Critical Introduction. London: Ro) | wikidata | `Q58150674`, a review of the 4th edition (2014) | none | No candidate is the book. |
| `linkedbook_6` (Cronachetta, Venezia, Visentini, 1880.) | wikidata | `Q125577424`, Goffi's *Cronachetta* (1979) | none | A different book. |

### 6c. Editions, translations and work items

The rules on reprints, translations and work items applied to every task where the
citation and the record name different printings of one work.

| Reference | Index | Was | Now | Reason |
|---|---|---|---|---|
| `linkedbook_80` (Shelley, *The Last Man*, London: Wordsworth, 1826, 2004) | wikidata | none | `Q2323961` | The work item of the novel (1826); nothing in it conflicts with the citation. |
| `linkedbook_69` (Cicognara, *Storia della scultura*, Prato: Giacchetti, 1823-25) | wikidata | `Q50658496`, the work item | `Q50660784` | The pool also holds the cited second edition (1823-24). |
| `excite_9543_39` (Lakatos, "Falsification and the Methodology of Scientific Research Programmes", 1970, in Worrall & Currie (eds.), pp. 8-101) | openalex | `W256019955`, the 1970 text in *Criticism and the Growth of Knowledge* (pp. 91-196) | `W1496810557` | The 1978 reprint in the volume edited by Worrall & Currie, pp. 8-101 as cited; only the date differs. `gold_doi` → `10.1017/cbo9780511621123.003`. |
| `excite_9543_39` | wikidata | `Q69801534`, the 1976 printing (pp. 205-259) | none | The pool holds only the 1970 and 1976 printings. |
| `brill_40375d98-6b48-450f-890c-b1447de69f8e_1155` (Wittgenstein, L. (tr. G. E. M. Anscombe) (1958) Philosophical Investigations. 2nd edn.) | wikidata | `Q772496`, the work item of the German original | none | The reference cites Anscombe's English translation; a translation is not the original's record. |

Checked and unchanged: `brill_a63e6f1f…_22` (von Albrecht, 2nd ed. 1994),
`linkedbook_64` (Garzoni 1585), `dasi_Muller2002`, `dasi_Nebes2004a` and
`ebda_Krebernik1996`, whose records are reprints or later digital versions that differ
only in date or carry the reference's own DOI; and Wikidata work items such as Durkheim
`Q588341` and Garzoni `Q19134406`.

### 6d. Records with the book's metadata and a review's DOI

Four OpenCitations records carry the cited book's own metadata (title, author, year,
publisher) but a review's DOI. The model never reads the DOI, so for it the record
is the book: the annotated link stands and `gold_doi` → null, since the review DOI is
not the book's. At serving time `/link/references?filter_reviews_by_doi=true` drops
candidates with a Choice Reviews DOI (`10.5860/choice.*`, reviews only); the other
three DOIs share their prefix with ordinary articles and cannot be filtered by pattern.

| Reference | Record | DOI |
|---|---|---|
| `10.1111_1467-6478.00057_instance-39` (Nedelsky, *Private Property…*, 1990) | `br/06019720685` | `10.5860/choice.29-0573` (*Choice*) |
| `10.1111_1467-6478.00057_instance-62` (Weitzman, *The Divorce Revolution*, 1985) | `br/06020652659` | `10.2307/352414` (*JMF* review) |
| `excite_18417_36` (SRU, *Umweltgutachten 1994*) | `br/06020122515` | `10.30965/25890581-07101011` (review article) |
| `excite_9543_59` (Todorov, *La conquête de l'Amérique*, 1982) | `br/06020237883` | `10.2307/2905909` (*MLN* review) |

### 6e. Tasks and references excluded

| Reference | Index | Reason |
|---|---|---|
| `brill_869d8a52-85bf-4144-b44e-bc5686aef65c_63` (Boeckh, *Die Staatshaushaltung der Athener*, 2nd ed. 1851) | Matilda, OpenAlex | Neither pool holds the cited edition; the 1886 edition, its "Band 1" and two 2010 reissues are equally plausible. Replaces the round-4 "no change" for Matilda. |
| `excite_9543_20` (Carr, *The Twenty Years' Crisis*, 1939/1964, Harper & Row) | OpenAlex | The 1946 book, a record without a year and a 2001 edition are equally plausible; none shows the cited Harper & Row printing. The accepted `W2022684055` was Griswold's 1941 review. |
| `10.1515_zfrs-1980-0104_instance-74` (Rokkan, *Vergleichende Sozialwissenschaft*, 1972, and others) | all | One footnote cites several works. |

### 6f. Other changes

- `excite_9543_34` (Katzenstein, *The Culture of National Security*): both OpenCitations
  candidates were identical records of the book; after deduplication the gold is
  `br/06021430190`. The annotated `br/06019117466` was never in the pool.
- Eight `none` labels flagged by the same scan stay `none`: `10.1111_1467-6478.00080_instance-14`
  (Huxley; pages and host volume differ), `excite_9543_46` (Melucci; the record is a
  chapter of another book), `cex_SOC-SCI_102_15` (the record's author is R. L. Jones,
  not J. M. Jones), and five whose candidate is a review (Euripides in Matilda, Martin
  in Matilda, SRU in Matilda and OpenAlex, Todorov in Matilda).
- `excite_9543_52` (Ringmar): `gold_doi` → `10.1017/cbo9780511557705`, the DOI all three gold
  records carry; the old value was the review's.
- `gold_doi` now comes only from confirmed matches that are still the gold after review:
  a match overridden as a review no longer contributes its DOI.

### 6g. Duplicate candidates

Providers, OpenCitations most, hold several records of one publication. Within one task,
candidates whose text as the model reads it (`linking_eval.format_record`) is identical
are collapsed to one: the gold if it is among them, else the higher-ranked; ranks are
renumbered. `build_report.json` (`review.removed_duplicate_candidates`) lists every
removed ID with the ID it duplicates: 192 candidates in 97 tasks (OpenCitations 38,
OpenAlex 33, Matilda 20, Wikidata 6); in 8 of them the gold had an identical twin, and
53 are `none` tasks.

