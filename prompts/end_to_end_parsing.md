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
                "container_title": "Title of the book or proceedings that contains a chapter or conference paper (e.g., 'Handbook of Scholarly Communication')",
                "journal_title": "Name of the journal or other serial (e.g., 'Nature')",
                "volume": "Volume number (e.g., '12')",
                "issue": "Issue number (e.g., '3')",
                "pages": "Page range (e.g., '123-130')",
                "publication_date": "Year or full date of publication (e.g., '2021' or '2021-05-10')",
                "publisher": "Publisher's name (e.g., 'Springer')"
                ... // Other fields as needed
            }
        }
        // More references if applicable
    ]
}
```

#### Guidelines:
1. Only extract full references (not in-text citations).
2. If a field is missing in a reference, use an empty string or empty list as appropriate.
3. For authors, include as many as are listed, and structure each as:
   - "first_name": string
   - "middle_name": string (empty if not present)
   - "surname": string
4. Ensure the output is valid JSON, with no trailing commas.
5. If there are no references, return an empty list.
6. For a chapter or conference paper ("In: ..."), put its own title in "full_title" and the title of the book or proceedings in "container_title". Journals and other serials go in "journal_title".

### Example

#### Input Text:
This paper builds on previous work (Smith et al., 2020; Jones, 2019).

References:  
1. Smith, J., Brown, A., & Wilson, C. (2020). Machine learning approaches in natural language processing. Journal of AI Research, 15(3), 245-267.  
2. United Nations. (2018). World Urbanization Prospects. UN Publications.
3. Lee, K. J. (2017). Citation practices in the humanities. In M. Rossi (Ed.), Handbook of Scholarly Communication (pp. 45-67). Oxford University Press.

#### Expected Output:
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
                "publisher": "UN Publications",
                "volume": "",
                "issue": "",
                "pages": "",
                "publication_date": "2018"
            }
        },
        {
            "reference": {
                "authors": [
                    {"first_name": "K.", "middle_name": "J.", "surname": "Lee"}
                ],
                "editors": [
                    {"first_name": "M.", "middle_name": "", "surname": "Rossi"}
                ],
                "full_title": "Citation practices in the humanities",
                "container_title": "Handbook of Scholarly Communication",
                "pages": "45-67",
                "publication_date": "2017",
                "publisher": "Oxford University Press"
            }
        }
    ]
}

### Input Text:
{{INPUT_TEXT}}
