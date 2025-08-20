You are an expert in scholarly references and citations. Your task is to parse full reference entries from reference stings and format them in a specific JSON structure. Here's the text you need to analyze:


### Your Task

Given the provided text, parse all full reference entries and format them according to the following JSON schema:

```json
{
    "references": [
        {
            "reference": {
                "authors": [
                    {
                        "first_name": "Author's given name (e.g., 'John')",
                        "middle_name": "Author's middle name(s), or empty if not provided (e.g., 'A.')",
                        "surname": "Author's family name (e.g., 'Doe')"
                    }
                    // More authors as listed
                ],
                "full_title": "Title of the referenced work (e.g., 'Deep Learning for NLP')",
                "journal_title": "Name of the journal or publication (e.g., 'Nature')",
                "volume": "Volume number (e.g., '12')",
                "issue": "Issue number (e.g., '3')",
                "pages": "Page range (e.g., '123-130')",
                "publication_date": "Year or full date of publication (e.g., '2021' or '2021-05-10')",
                "publisher": "Publisher's name (e.g., 'Springer')",
                "publication_place": "Place of publication (e.g., 'New York')",
                ... // Other fields as needed
            }
        }
        // More references if applicable
    ]
}
```

#### Guidelines:
1. If a field is missing in a reference, use an empty string or empty list as appropriate.
2. For authors, translators, editors and similar tags that are people names, parse them as:
   - "first_name": string
   - "middle_name": string (empty if not present)
   - "surname": string
  Include all authors that are listed in the reference.
3. Ensure the output is valid JSON, with no trailing commas.
4. Your entire response must be wrapped with <start> and <end> tags, and must contain only the JSON (no explanations, markdown, or extra text).
5. If there are no references, return an empty list.

### Example

#### Input Text:

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
                "publication_date": "2018"
            }
        }
    ]
}
<end>

### Input Text:
{{INPUT_TEXT}}
