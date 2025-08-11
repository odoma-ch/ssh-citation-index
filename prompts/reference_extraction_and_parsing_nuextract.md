# Reference Extraction and Parsing - NuExtract Template

This prompt is designed for use with NuExtract models. Provide the input text and use the following JSON template to extract and parse reference information in a single step.

## JSON Template

```json
{
  "references": [
    {
      "reference": {
        "authors": [
          {
            "first_name": "verbatim-string",
            "middle_name": "verbatim-string",
            "surname": "verbatim-string"
          }
        ],
        "organization": "verbatim-string",
        "full_title": "verbatim-string",
        "journal_title": "verbatim-string",
        "volume": "integer",
        "issue": "integer",
        "pages": "verbatim-string",
        "publication_date": "date-time",
        "publisher": "verbatim-string"
      }
    }
  ]
}
```

## Template Explanation

- `references`: An array containing all extracted and parsed references
- `reference`: Object containing the structured reference data
- `authors`: Array of author objects (use empty array if no authors)
- `first_name`: Author's given name (verbatim from source)
- `middle_name`: Author's middle name(s) or empty string if not provided (verbatim from source)
- `surname`: Author's family name (verbatim from source)
- `organization`: Organization name (for institutional authors, use empty string if not applicable)
- `full_title`: Complete title of the referenced work (verbatim from source)
- `journal_title`: Name of the journal or publication (verbatim from source)
- `volume`: Volume number (integer)
- `issue`: Issue number (integer)
- `pages`: Page range (verbatim from source, e.g., "123-130")
- `publication_date`: Year or full date of publication (ISO date format)
- `publisher`: Publisher's name (verbatim from source)

## Expected Output Format

The model will return a JSON object with an array of structured references, where each reference contains parsed components extracted from the input document.

## Example Usage

### Input Text:
```
This paper builds on previous work (Smith et al., 2020; Jones, 2019).

References:
1. Smith, J., Brown, A., & Wilson, C. (2020). Machine learning approaches in natural language processing. Journal of AI Research, 15(3), 245-267.
2. United Nations. (2018). World Urbanization Prospects. UN Publications.
```

### Expected Output:
```json
{
  "references": [
    {
      "reference": {
        "authors": [
          {"first_name": "J.", "middle_name": "", "surname": "Smith"},
          {"first_name": "A.", "middle_name": "", "surname": "Brown"},
          {"first_name": "C.", "middle_name": "", "surname": "Wilson"}
        ],
        "organization": "",
        "full_title": "Machine learning approaches in natural language processing",
        "journal_title": "Journal of AI Research",
        "volume": "15",
        "issue": "3",
        "pages": "245-267",
        "publication_date": "2020",
        "publisher": ""
      }
    },
    {
      "reference": {
        "authors": [],
        "organization": "United Nations",
        "full_title": "World Urbanization Prospects",
        "journal_title": "UN Publications",
        "volume": "",
        "issue": "",
        "pages": "",
        "publication_date": "2018",
        "publisher": ""
      }
    }
  ]
}
```

## Notes

- Use temperature at or very close to 0 for best results
- The model will extract only full reference entries, not in-text citations
- References are typically found at the end of documents under headings like "References," "Bibliography," or "Works Cited"
- If a field is missing in the source reference, use an empty string
- For institutional authors (like "United Nations"), use the `organization` field and leave `authors` as an empty array
- This template combines both extraction (finding references in the document) and parsing (structuring the reference data)
- If no references are found, the model will return an empty array

# Input Text:
{{INPUT_TEXT}}