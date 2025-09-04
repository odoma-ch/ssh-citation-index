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
1. Ensure the output is valid JSON, with no trailing commasl, and must contain only the JSON (no explanations, markdown, or extra text).
2. If there are no references, return an empty list.

### Examples
#### Input Text:
Archivio ( L ’) di Stato di Venezia negli anni 1878 - 1880, Vene - zia 1881.


#### Expected Output:

{
    "references": [
        {
            "reference": {
                "title": "Archivio ( L ’) di Stato di Venezia negli anni 1878 - 1880,",
                "publicationplace": "Vene - zia",
                "year": "1881."
            }
        }
    ]
}

#### Input Text:
1. Bomford D., G. Finaldi, Venice through Canaletto ’ s Eyes, London 1998.
2. Baart, Jan. “ Dutch material civilization: daily life between 1650 - 1776. Evidence from archaeology.” In New World Dutch Studies: Dutch Arts and Culute in Colonial America 1609 - 1776, edited by R. Blackburn and N. Kelley, 1 - 11. Albany: Albany Institute of History and Art, 1987.

#### Expected Output:

{
    "references": [
        {
            "reference": {
                "authors": "Bomford D., G. Finaldi,",
                "full_title": "Venice through Canaletto ’ s Eyes,",
                "publicationplace": "London",
                "year": "1998."
            }
        },
        {
            "reference": {
                "authors": "Baart, Jan., edited by R. Blackburn and N. Kelley",
                "full_title": "“ Dutch material civilization: daily life between 1650 - 1776. Evidence from archaeology.” New World Dutch Studies: Dutch Arts and Culute in Colonial America 1609 - 1776,",
                "conjunction": "In",
                "pagination": "1 - 11.",
                "publicationplace": "Albany:",
                "publisher": "Albany Institute of History and Art,",
                "year": "1987."
            }
        },
    ]
}


### Input Text:
{{INPUT_TEXT}}
