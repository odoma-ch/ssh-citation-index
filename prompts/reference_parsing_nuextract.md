# Template

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

# Context:
# Reference Parsing - NuExtract Template

This prompt is designed for use with NuExtract models. Provide the reference text and use the following JSON template to parse and structure the reference information.


## Expected Output Format

The model will return a JSON object with an array of structured references, where each reference contains parsed components extracted from the input reference text.

## Example Usage

### Input Text:
```
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
- If a field is missing in the source reference, use an empty string
- For institutional authors (like "United Nations"), use the `organization` field and leave `authors` as an empty array
- The model will parse the reference text and extract structured information according to the template
- Ensure all string fields are properly quoted in the JSON output

# Input Text:
{{INPUT_TEXT}}