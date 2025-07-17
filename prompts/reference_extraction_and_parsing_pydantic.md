You are an expert in scholarly references and citations. Your task is to extract all full reference entries from scientific works and format them in a specific JSON structure. Here's the text you need to analyze:


### Where to Find References

References may appear in the following locations:
- At the end of the document, under headings such as "References," "Bibliography," or "Works Cited."
- In footnotes at the bottom of pages, or as endnotes at the end of the document or chapter.
- Occasionally, in other sections such as appendices or figure/table captions.

**Do not extract in-text citations (e.g., "(Smith et al., 2020)") unless they are accompanied by a full reference entry.**

### Your Task

Given the provided text, extract all full reference entries and format them according to the following JSON schema:

```json
{{JSON_SCHEMA_FOR_REFERENCES_WRAPPER}}
```

#### Guidelines:
1. Only extract full references (not in-text citations).
2. If a field is missing in a reference, use an empty string or empty list as appropriate.
3. For authors, include as many as are listed, and structure each as:
   - "first_name": string
   - "middle_name": string (empty if not present)
   - "surname": string
4. If the author is an organization, use the "surname" field and leave "first_name" and "middle_name" empty.
5. Ensure the output is valid JSON, with no trailing commas.
6. Your entire response must be wrapped with <start> and <end> tags, and must contain only the JSON (no explanations, markdown, or extra text).

### Example

#### Input Text:
This paper builds on previous work (Smith et al., 2020; Jones, 2019).

References:  
1. Smith, J., Brown, A., & Wilson, C. (2020). Machine learning approaches in natural language processing. Journal of AI Research, 15(3), 245-267.  
2. United Nations. (2018). World Urbanization Prospects. UN Publications.

#### Expected Output:
<start>
{
    "references": [
        {
            "reference": {
                "authors": [
                    {"first_name": "J.", "middle_name": "", "surname": "Smith"},
                    {"first_name": "A.", "middle_name": "", "surname": "Brown"},
                    {"first_name": "C.", "middle_name": "", "surname": "Wilson"}
                ],
                "full_title": "Machine learning approaches in natural language processing",
                "journal_title": "Journal of AI Research",
                "volume": "15",
                "issue": "3",
                "pages": "245-267",
                "publication_date": "2020"
            }
        },
        {
            "reference": {
                "organization": "United Nations",
                "full_title": "World Urbanization Prospects",
                "journal_title": "UN Publications",
                "volume": "",
                "issue": "",
                "pages": "",
                "publication_date": "2018"
            }
        }
    ]
}
<end>

### Input Text:
{{INPUT_TEXT}}
