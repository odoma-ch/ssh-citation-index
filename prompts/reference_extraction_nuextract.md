# Reference Extraction - NuExtract Template

This prompt is designed for use with NuExtract models. Provide the input text and use the following JSON template to extract reference information.

## JSON Template

```json
{
  "references": [
    {
      "reference_text": "verbatim-string"
    }
  ]
}
```

## Template Explanation

- `references`: An array containing all extracted references
- `reference_text`: The full text of each reference as it appears in the source document

## Expected Output Format

The model will return a JSON object with an array of references, where each reference contains the verbatim text extracted from the input document.

## Example Usage

### Input Text:
```
This paper builds on previous work (Smith et al., 2020; Jones, 2019). According to recent studies...

References:
1. Smith, J., Brown, A., & Wilson, C. (2020). Machine learning approaches in natural language processing. Journal of AI Research, 15(3), 245-267.
2. Jones, M. (2019). Deep learning fundamentals. MIT Press.
3. Davis, R., & Lee, S. (2021). Neural networks and their applications. Nature Machine Intelligence, 3(2), 112-125.
```

### Expected Output:
```json
{
  "references": [
    {
      "reference_text": "Smith, J., Brown, A., & Wilson, C. (2020). Machine learning approaches in natural language processing. Journal of AI Research, 15(3), 245-267."
    },
    {
      "reference_text": "Jones, M. (2019). Deep learning fundamentals. MIT Press."
    },
    {
      "reference_text": "Davis, R., & Lee, S. (2021). Neural networks and their applications. Nature Machine Intelligence, 3(2), 112-125."
    }
  ]
}
```

## Notes

- Use temperature at or very close to 0 for best results
- The model will extract only full reference entries, not in-text citations
- References are typically found at the end of documents under headings like "References," "Bibliography," or "Works Cited"
- If no references are found, the model will return an empty array

# Input Text:
{{INPUT_TEXT}}