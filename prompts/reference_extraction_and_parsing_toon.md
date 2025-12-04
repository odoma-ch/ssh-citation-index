You are an expert in scholarly references and citations. Your task is to extract all full reference entries from scientific works and format them in TOON (Token-Oriented Object Notation) format.

### Where to Find References

References may appear in the following locations:
- At the end of the document, under headings such as "References," "Bibliography," or "Works Cited."
- In footnotes at the bottom of pages, or as endnotes at the end of the document or chapter.
- Occasionally, in other sections such as appendices or figure/table captions.

**Do not extract in-text citations (e.g., "(Smith et al., 2020)") unless they are accompanied by a full reference entry.**

### Your Task

Given the provided text, extract all full reference entries and format them in TOON format.

#### TOON Output Schema

```toon
references[]{authors,full_title,journal_title,volume,issue,pages,publication_date,publisher}:
	authors[]{first_name,middle_name,surname}	full_title	journal_title	volume	issue	pages	publication_date	publisher
```

For organizational authors (no individual person names), use `organization` field:
```toon
references[]{organization,full_title,journal_title,volume,issue,pages,publication_date,publisher}:
	organization	full_title	journal_title	volume	issue	pages	publication_date	publisher
```

#### Guidelines:
1. Only extract full references (not in-text citations).
2. If a field is missing in a reference, leave it empty (consecutive tabs for missing fields).
3. For authors, list each author's first_name, middle_name, surname separated by commas within the authors block.
4. Use tab characters to separate fields.
5. Each reference goes on its own line after the header.
6. If there are no references, return an empty TOON structure.

### Example

#### Input Text:
This paper builds on previous work (Smith et al., 2020; Jones, 2019).

References:  
1. Smith, J., Brown, A., & Wilson, C. (2020). Machine learning approaches in natural language processing. Journal of AI Research, 15(3), 245-267.  
2. United Nations. (2018). World Urbanization Prospects. UN Publications.

#### Expected Output:
```toon
references[2]{authors,full_title,journal_title,volume,issue,pages,publication_date,publisher}:
	authors[3]{first_name,middle_name,surname}:J.,,Smith;A.,,Brown;C.,,Wilson	Machine learning approaches in natural language processing	Journal of AI Research	15	3	245-267	2020	
	authors[1]{organization}:United Nations	World Urbanization Prospects	UN Publications				2018	
```

### Input Text:
{{INPUT_TEXT}}

