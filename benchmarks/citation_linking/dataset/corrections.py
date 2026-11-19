"""Every reviewed correction of the citation-linking gold, applied by ``build_annotation_dataset.py``.

Data only. ``CORRECTIONS.md`` gives the reason for each entry, round by round:

- ``ANNOTATION_FIXES``: one contradictory Argilla answer replaced; the label is derived again.
- ``EXCLUDED_REFERENCES``: whole references removed from the release.
- ``EXCLUDED_TASKS``: single ``(ref_id, index)`` tasks removed from the release.
- ``REVIEW_OVERRIDES``: gold IDs (per index) and gold DOIs that replace the annotation evidence.

These are dict literals: a repeated key silently keeps only the last entry, so merge
corrections of one reference into one entry (``ruff check --select F601`` catches repeats).
"""

# Contradictory or incomplete answers, corrected after review. Each entry replaces one
# answer of one (ref_id, index) annotation; the label is derived again from the answers.
# (ref_id, index): (question, corrected answer, what changed and why)
ANNOTATION_FIXES = {
    ("brill_a15e3650-8e13-4550-8b67-2158fc2df786_634", "openalex"): (
        "no_match",
        "true",
        "no_match: false -> true (no candidate returned and no_match was left false; the matilda and wikidata rows for this reference both read correctly-not-found, so this is a correct abstention)",
    ),
    ("10.1515_zfrs-1980-0104_instance-63", "openalex"): (
        "no_match",
        "true",
        "no_match: false -> true (annotator marked the record as present in OpenAlex but gave no ID; OpenAlex holds 91 Marshall Clinard works, none on victimization, and no title match for 'Comparative Crime Victimization Surveys' 1978 -- the work is absent, so the returned candidate is a spurious link)",
    ),
    ("cex_COM-SCI_28_20", "matilda"): (
        "is_match_correct",
        "false",
        "is_match_correct: true -> false (candidate 'Pixeldust Studios Reptopia Magic Leap Experience' is a different work from the reference 'Magic Leap. 2018'; matches how openalex was answered)",
    ),
    ("10.1111_1467-6478.00057_instance-71", "matilda"): (
        "correct_id",
        "[SKIP] multiple references packed in one single context (Harvard Law Review eds. 1990; Dean v. District of Columbia 1995; Overington, The Age 1997)",
        "correct_id -> [SKIP] (reference text bundles three publications, the same class as the 18 references the openalex annotator marked; excluded from the analysis)",
    ),
    ("10.1515_zfrs-1980-0104_instance-7", "matilda"): (
        "is_match_correct",
        "false",
        "is_match_correct: true -> false (candidate is an unrelated PSSRI volume on the China-Pakistan Economic Corridor)",
    ),
    ("cex_COM-SCI_28_20", "wikidata"): (
        "is_match_correct",
        "false",
        "is_match_correct: true -> false (candidate Q18351488 is Magic Leap the company, not a publication record)",
    ),
    ("brill_7b98252c-c219-491b-8a44-3e4286c9ca47_2", "opencitations"): (
        "is_match_correct",
        "false",
        "is_match_correct: true -> false (candidate br/06020233318 is Barco, a 2011 catalogue of the BnF Hebrew manuscripts 1-32, not the cited Latin manuscript 'ms lat. 8589'; annotator also marked no_match=true and the openalex/matilda/wikidata rows for this reference read correctly-not-found, so this is a spurious link)",
    ),
    ("dasi_Brunner2001", "openalex"): (
        "is_match_correct",
        "false",
        "is_match_correct: true -> false (candidate W2750627499 is Linke, 'Gender - Bilder - Sanaa' (2017), an unrelated work; annotator also marked no_match=true and the matilda/wikidata rows for this reference read correctly-not-found, so this is a spurious link)",
    ),
}
# Reviewed corrections (2026-10, 2026-11) that override raw annotation evidence.
# gold_id entries are keyed by target index. cex_ENV-SCI_54_63 needs no entry:
# both DOIs are valid, so its gold_doi stays null and the conflict stays listed.
# References removed from the release after review, with the reason.
EXCLUDED_REFERENCES = {
    # Authors and year are Harvey & Rauch 1997 (Health & Social Work 22); title,
    # journal and volume are Harvey & Hill 2004 (Social Work 49). The cited work
    # cannot be told, and both papers are indexed.
    "cex_SOC-SCI_102_13": "reference_blends_two_works",
    # One footnote citing several works (Rokkan 1972; Szalai; IDE group).
    "10.1515_zfrs-1980-0104_instance-74": "multi_ref_context",
}
# Single (ref_id, index) tasks removed after review (round 6). The label cannot be
# decided from the citation and the candidate texts alone.
EXCLUDED_TASKS = {
    # Cited: 2nd ed. 1851 (2 vols). Pools: 1886 3rd ed., its "Band 1", 2010 reprints;
    # several are equally plausible and none is the cited edition.
    (
        "brill_869d8a52-85bf-4144-b44e-bc5686aef65c_63",
        "matilda",
    ): "several_editions_equally_plausible",
    (
        "brill_869d8a52-85bf-4144-b44e-bc5686aef65c_63",
        "openalex",
    ): "several_editions_equally_plausible",
    # Cited: 1939/1964, Harper & Row. Pool: 1946 book, a record without year, 2001 ed.
    ("excite_9543_20", "openalex"): "several_editions_equally_plausible",
}
REVIEW_OVERRIDES = {
    "10.1111_1467-6478.00057_instance-38": {"gold_id": {"openalex": None}},
    "excite_32211_33": {"gold_id": {"openalex": None}},
    "linkedbook_26": {"gold_id": {"openalex": None}},
    # Dead ID; the Italian translation the reference cites was found again (2026-11).
    "linkedbook_5": {"gold_id": {"openalex": "W618320456"}},
    "linkedbook_73": {"gold_id": {"openalex": None}},
    "linkedbook_94": {"gold_id": {"openalex": "W7019729045"}},
    "cex_PHY-AST_96_10": {
        "gold_id": {"openalex": "W1968542656"},
    },
    # Round 6: Matilda holds the book; the OpenCitations record carries the DOI of a
    # Contemporary Sociology review, br/062403046395 is the book.
    "excite_9543_52": {
        # All three gold records carry the book's DOI; the evidence had the review's.
        "gold_doi": "10.1017/cbo9780511557705",
        "gold_id": {
            "openalex": "W4233971661",
            "matilda": "faaaee23-6c67-46f3-a4f8-01efb0d11aeb",
            "opencitations": "br/062403046395",
        },
    },
    # Same book under two DOIs; keep the older record (original 1989 date).
    "10.1111_1467-6478.00057_instance-63": {
        "gold_doi": "10.1515/9780691215983",
        "gold_id": {"matilda": "d6313d68-4fc7-4307-9b92-48b25363cdda"},
    },
    # OpenAlex is right; the Matilda record is a review of the book by another author.
    "brill_20f2caaa-677a-42c1-8857-19e196f8c74e_331": {
        "gold_doi": "10.1017/cbo9780511583599",
        "gold_id": {"matilda": None},
    },
    # The cited work is the 1996 book chapter (Matilda, OpenAlex W4299787999). The
    # OpenCitations and Wikidata records are the 1994 journal version of the same text.
    "excite_9543_36": {
        "gold_doi": "10.1515/9781685853952-010",
        "gold_id": {
            "openalex": "W4299787999",
            "opencitations": None,
            "wikidata": None,
        },
    },
    # The cited work is a book chapter; OpenAlex had matched the whole book.
    "excite_9543_10": {
        "gold_doi": "10.7560/720466-008",
        "gold_id": {"openalex": "W4408028697"},
    },
    # Cross-index review after the OpenCitations round (2026-11): references where
    # annotators judged the same DOI differently. A gold record is the cited
    # publication itself, not a review of it or another edition of the same text.
    # Rejected candidates that the annotator's index does hold:
    "excite_9543_33": {"gold_id": {"openalex": "W58313853"}},
    "10.1515_zfrs-1980-0103_instance-30": {"gold_id": {"openalex": "W181794323"}},
    # OpenAlex had accepted a different paper (Korhonen et al., Ecological Economics).
    "cex_ENV-SCI_54_70": {
        "gold_doi": "10.1016/j.resconrec.2017.08.027",
        # Round 6: Wikidata item of the same article, differing only in date.
        "gold_id": {"openalex": "W2769483102", "wikidata": "Q125841968"},
    },
    # Book records that carry the DOI of a book review: the gold IDs stand, the DOI
    # does not identify the cited book.
    # Round 6: both OpenCitations pool candidates are the book (identical records,
    # deduplicated to br/06021430190); the annotated br/06019117466 was never in the pool.
    "excite_9543_34": {
        "gold_doi": None,
        "gold_id": {"opencitations": "br/06021430190"},
    },
    "10.1111_1467-6478.00057_instance-40": {"gold_doi": None},
    # 1924 four-page article of the same title, not the 1926 book (Wikidata has it).
    "linkedbook_85": {"gold_doi": None, "gold_id": {"opencitations": None}},
    # Reviews of the cited book (by Drinan and by Foster), not the book.
    "10.1515_zfrs-1980-0104_instance-43": {
        "gold_doi": None,
        # Round 6: OpenAlex holds the book (record added after annotation).
        "gold_id": {
            "openalex": "W6983264735",
            "opencitations": None,
            "wikidata": None,
        },
    },
    # Harvey & Hill 2004: same title and journal, different co-author and year.
    # DASI/EBDA review (2026-11): these references print their DOI, which was looked
    # up in each index. OpenAlex merged the annotated record into another one.
    "dasi_Yule2002": {"gold_id": {"openalex": "W3112109909"}},
    # The accepted OpenCitations record is another article by Wellsted in the same
    # volume (p. 102, DOI 10.2307/1797516); the cited DOI is not in OpenCitations.
    "dasi_Wellsted1837": {
        "gold_doi": "10.2307/1797510",
        "gold_id": {"opencitations": None},
    },
    # Matilda holds the article (found by its printed DOI); no candidate was shown.
    "ebda_DalleyTeissier1992": {
        "gold_id": {"matilda": "7847d2f3-385d-497b-a716-8e030d4d3a56"}
    },
    # Round 6 (2026-11): every label checked against the frozen candidate pool
    # (CORRECTIONS.md, round 6). A label must follow from the citation and the
    # candidate texts. `none` answers whose record is in the pool: the record entered
    # the index after annotation, the annotator's search missed it, or it differs
    # only in date (a reprint) or pages under the same DOI.
    "10.1111_1467-6478.00057_instance-41": {"gold_id": {"openalex": "W7062005228"}},
    "10.1111_1467-6478.00080_instance-18": {"gold_id": {"openalex": "W2970212067"}},
    "10.1111_1467-6478.00080_instance-48": {
        "gold_id": {"matilda": "05a57523-0145-4be5-9d53-d7424530def4"}
    },
    "10.1515_zfrs-1980-0104_instance-15": {
        "gold_id": {"matilda": "9e0e05af-eda6-4c45-86de-8a9caeedeec3"}
    },
    "10.1515_zfrs-1980-0104_instance-19": {"gold_id": {"openalex": "W7071678455"}},
    "10.1515_zfrs-1980-0104_instance-33": {"gold_id": {"openalex": "W7006124636"}},
    "10.1515_zfrs-1980-0104_instance-66": {
        "gold_id": {"matilda": "f467924c-f203-46c6-bbb1-bc6082de0817"}
    },
    "brill_275e72b8-c496-4d90-959e-40657ca53810_134": {
        "gold_id": {"openalex": "W7204422925"}
    },
    "brill_4c6ba321-d757-4ab1-b4e9-568fd2087e41_285": {
        "gold_id": {"openalex": "W4210995859"}
    },
    "brill_5b7ade43-7b83-4597-a458-042d8a99632f_64": {
        "gold_id": {"openalex": "W1970677847"}
    },
    "brill_624697a1-c86b-40fb-91d0-8a89d95dc958_138": {
        "gold_id": {"openalex": "W7070867014"}
    },
    "brill_81977646-2ce4-4c77-8ee6-0ef78a9cd358_588": {
        "gold_id": {"openalex": "W7073812697"}
    },
    "brill_969dc0be-9925-4ee3-a7ec-c98285b071f6_484": {
        "gold_id": {"matilda": "44ed92bb-6417-4921-9b40-028e29529880"}
    },
    "brill_a63e6f1f-cf2e-44e5-bab5-0709ecf2fc82_22": {
        "gold_id": {"matilda": "da689b31-3d6c-4317-b1e3-7546701098b1"}
    },
    "cex_COM-SCI_28_29": {"gold_id": {"openalex": "W7028184114"}},
    "cex_DEN_35_15": {"gold_id": {"openalex": "W2397755785"}},
    "cex_ENV-SCI_54_29": {"gold_id": {"opencitations": "br/06019323514"}},
    "cex_ENV-SCI_54_31": {"gold_id": {"wikidata": "Q57551678"}},
    "cex_ENV-SCI_54_65": {"gold_id": {"openalex": "W2761541636"}},
    "cex_PHY-AST_96_14": {
        "gold_id": {"matilda": "2e526547-5a7d-4142-b9ed-2b7526059013"}
    },
    "dasi_Edens1999": {
        "gold_id": {
            "matilda": "496d5fc1-4c2b-4e40-85fe-e5e336fbc02c",
            "wikidata": "Q57476809",
        }
    },
    "dasi_Robin2001": {"gold_id": {"matilda": "463f02a8-d09b-4992-b784-3a9782196e22"}},
    "dasi_Stein2003a": {"gold_id": {"openalex": "W7062438958"}},
    "ebda_AscalonePeyronel2000": {"gold_id": {"openalex": "W7037298554"}},
    "ebda_Barstad1978": {"gold_id": {"openalex": "W1541528659"}},
    "ebda_Biga2014d": {
        "gold_id": {"openalex": "W2492327343", "wikidata": "Q127150057"}
    },
    "ebda_Dahood1982a": {"gold_id": {"openalex": "W7070538371"}},
    "ebda_Dolce2025": {
        "gold_id": {
            "matilda": "015dde3d-980e-41fb-851e-abd098e9b7a1",
            "openalex": "W7140359981",
        }
    },
    "ebda_Fales1984a": {"gold_id": {"openalex": "W7029531406"}},
    "ebda_Matthiae1969": {
        "gold_id": {"matilda": "8fb0ce6b-96bb-42dc-aaee-297528f6f4aa"}
    },
    "ebda_Matthiae1982": {"gold_id": {"openalex": "W7046354894"}},
    "ebda_Matthiae2019d": {
        "gold_id": {
            "matilda": "88728b69-4e79-4304-b639-08bd9dd2c6db",
            "openalex": "W4402107120",
        }
    },
    "ebda_Micale2018": {"gold_id": {"matilda": "08db4245-1356-4f01-86cc-5d1e0286169f"}},
    "ebda_Pinnock1979": {"gold_id": {"openalex": "W7030440895"}},
    "excite_18417_10": {
        "gold_id": {
            "matilda": "05ecb5f1-d2e8-4976-912c-842eacf5f876",
            "openalex": "W4239222949",
        }
    },
    "excite_18417_31": {"gold_id": {"openalex": "W7016054603"}},
    "excite_32211_14": {
        "gold_id": {
            "matilda": "99d5aa2d-3fb2-4a47-9ab2-1ec0e6090d3c",
            "openalex": "W2565235294",
        }
    },
    "excite_9543_0": {"gold_id": {"matilda": "c50d9fcd-de90-45c3-8a06-f8ac54c06a4c"}},
    "excite_9543_14": {"gold_id": {"matilda": "607b19cd-71e0-4374-8020-a1700ed6579d"}},
    "excite_9543_23": {"gold_id": {"wikidata": "Q139686849"}},
    "linkedbook_105": {"gold_id": {"openalex": "W619234676"}},
    "linkedbook_99": {"gold_id": {"openalex": "W7033185218"}},
    # Accepted records that are a review of the cited book or another work; where
    # the pool holds the book itself (or a reprint differing only in date), it is
    # the gold. Wittgenstein: the item is the German original, not the cited
    # English translation.
    # OpenCitations records whose metadata is the cited book (title, author, year) but
    # whose DOI is a review's. The model never sees the DOI, so the link stands;
    # ``filter_reviews_by_doi`` drops Choice DOIs at serving time. The review DOI is
    # not the book's, so gold_doi is null.
    "10.1111_1467-6478.00057_instance-39": {
        "gold_doi": None,
        "gold_id": {"openalex": None},
    },
    "10.1111_1467-6478.00057_instance-62": {
        "gold_doi": None,
        "gold_id": {"wikidata": None},
    },
    "excite_18417_36": {"gold_doi": None},
    "excite_9543_59": {"gold_doi": None},
    "brill_0172ea14-22fd-4164-a417-cb923c77acb3_588": {
        "gold_id": {"matilda": "d53289af-4344-4464-b144-6216065c7b30", "wikidata": None}
    },
    "brill_2841ce8b-9934-4665-8c6c-fc3fca5b31a4_48": {"gold_id": {"wikidata": None}},
    "brill_40375d98-6b48-450f-890c-b1447de69f8e_1155": {"gold_id": {"wikidata": None}},
    "brill_591b2204-041d-4d3f-937f-8ef19f7f73bf_214": {
        "gold_id": {"openalex": "W431884289"}
    },
    "brill_63aad500-2bf4-4cb4-ae2b-14a84d24bef3_1126": {"gold_id": {"wikidata": None}},
    "brill_c90e527d-b3cc-4ae0-b2c5-f959ba70584e_106": {
        "gold_id": {"openalex": "W4236503687"}
    },
    "brill_f288353a-0a2e-4095-a902-9115298934f4_298": {"gold_id": {"wikidata": None}},
    "cex_ECO-ECO-FIN_41_14": {"gold_id": {"wikidata": None}},
    "cex_SOC-SCI_102_20": {"gold_id": {"openalex": None}},
    "excite_9543_20": {"gold_id": {"wikidata": None}},
    "excite_9543_58": {"gold_id": {"wikidata": None}},
    "excite_9543_62": {"gold_id": {"wikidata": None}},
    "linkedbook_6": {"gold_id": {"wikidata": None}},
    # Editions and work items. A Wikidata work item is a valid gold when nothing in the
    # citation or the item conflicts; where the pool holds the cited edition, that
    # edition is the gold.
    "linkedbook_80": {"gold_id": {"wikidata": "Q2323961"}},  # The Last Man (work)
    "linkedbook_69": {"gold_id": {"wikidata": "Q50660784"}},  # 2nd ed. 1823-24
    # Lakatos, "Falsification…", cited in Worrall & Currie (eds.), pp. 8-101: the 1978
    # Philosophical Papers reprint (only the date differs from the cited 1970). The
    # OpenAlex gold was the 1970 Criticism and the Growth of Knowledge text (pp. 91-196);
    # Wikidata holds only the 1970 and 1976 printings.
    "excite_9543_39": {
        "gold_doi": "10.1017/cbo9780511621123.003",
        "gold_id": {"openalex": "W1496810557", "wikidata": None},
    },
}
