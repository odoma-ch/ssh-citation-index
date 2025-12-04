You are an expert in scholarly references and citations. Your task is to parse full reference entries from reference strings and format them in TOON (Token-Oriented Object Notation) format.

### Your Task

Given the provided text, parse all full reference entries and format them in TOON format.

#### TOON Output Schema

```toon
references[]{authors,full_title,journal_title,volume,issue,pages,publication_date,publisher,publication_place,identifier}:
	authors[]{first_name,middle_name,surname}	full_title	journal_title	volume	issue	pages	publication_date	publisher	publication_place	identifier
```

For organizational authors, use:
```toon
references[]{organization,full_title,journal_title,volume,issue,pages,publication_date,publisher,publication_place,identifier}:
	organization	full_title	journal_title	volume	issue	pages	publication_date	publisher	publication_place	identifier
```

#### Guidelines:
1. If a field is missing in a reference, leave it empty (consecutive tabs for missing fields).
2. For authors, translators, editors and similar person names, parse as:
   - first_name, middle_name (empty if not present), surname
   - Separate multiple authors with semicolons within the authors block.
3. Use tab characters to separate fields.
4. Each reference goes on its own line after the header.
5. Output must contain only the TOON data (no explanations, markdown, or extra text).
6. If there are no references, return an empty TOON structure.

### Example

#### Input Text:

1. Smith, J., Brown, A., & Wilson, C. (2020). Machine learning approaches in natural language processing. Journal of AI Research, 15(3), 245-267.  
2. United Nations. (2018). World Urbanization Prospects. UN Publications.

#### Expected Output:
```toon
references[2]{authors,full_title,journal_title,volume,issue,pages,publication_date,publisher,publication_place,identifier}:
	authors[3]{first_name,middle_name,surname}:J.,,Smith;A.,,Brown;C.,,Wilson	Machine learning approaches in natural language processing	Journal of AI Research	15	3	245-267	2020			
	authors[1]{organization}:United Nations	World Urbanization Prospects	UN Publications				2018			
```

### Input Text:
{{INPUT_TEXT}}

