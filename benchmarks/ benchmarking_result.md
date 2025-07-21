## Targeting model
- Deepseek v3 (API)
- Claude 4 Sonnet (API)
- Gemma 3 27b (PSNC VM)
- mistralai/Mistral-Small-3.2-24B-Instruct-2506 (PSNC VM)
- 


### EXCITE dataset

Total documents: TODO
Total references: TODO


reference extraction :

Here the failure reason mainly because the pdf is too large for the context window.

| LLM                                      | P    | R    | F1   | Similarity   | # Failed Docs | runtime (s) |
|------------------------------------------|------|------|------|--------------|---------------|-------------|
| Deepseek v3                              | 0.7272 | 0.7375 | 0.7229 | 0.7848 | 13          | TODO        |
| Claude 4 Sonnet                          | TODO | TODO | TODO | TODO         | TODO          | TODO        |
| Gemma 3 27b                              | TODO | TODO | TODO | TODO         | TODO          | TODO        |
| Mistral-Small-3.2-24B-Instruct-2506      | TODO | TODO | TODO | TODO         | TODO          | TODO        |


| LLM                                      | P    | R    | micro F1   | Marco F1 | # Failed Docs | runtime (s) | Title F1 | Author F1 | publication year F1 |
|------------------------------------------|------|------|------|--------------|-------------|----------|----------|--------------------|
| Deepseek v3  (API)                       | 0.8207 | 0.7789 | 0.7817 | 0.7594 | 6 |1612.73 s   | TODO     | TODO     | TODO               |
| Claude 4 Sonnet                          | TODO | TODO | TODO | TODO         | -           | TODO     | TODO     | TODO               |
| Gemma 3 27b                              | 0.8102 | 0.8094 | 0.8054 | 0.7879 | 10 | 2245.3 s    | TODO     | TODO     | TODO               |
| Mistral-Small-3.2-24B-Instruct-2506      | 0.7923 | 0.7985 | 0.7928 | 0.7770 | 20 | 1421.17 s   | TODO     | TODO     | TODO               |


{
  "precision": 0.7922778097982708,
  "recall": 0.7984573487031701,
  "micro_f1": 0.7927835734870317,
  "macro_f1": 0.7770432276657061
}


reference extraction and parsing :

| LLM                                  | P    | R    | F1   | Similarity   | # Failed Docs | runtime (s) | Title F1 | Author F1 | publication year F1 |
|------------------------------------------|------|------|------|--------------|---------------|-------------|----------|----------|--------------------|
| Deepseek v3                              | TODO | TODO | TODO | TODO         | TODO          | TODO        | TODO     | TODO     | TODO               |
| Claude 4 Sonnet                          | TODO | TODO | TODO | TODO         | TODO          | -           | TODO     | TODO     | TODO               |
| Gemma 3 27b                              | TODO | TODO | TODO | TODO         | TODO          | 3503.94 s   | TODO     | TODO     | TODO               |
| Mistral-Small-3.2-24B-Instruct-2506      | TODO | TODO | TODO | TODO         | TODO          | 5817.9s     | TODO     | TODO     | TODO               |



## CEXgoldstandard dataset

Total documents: TODO
Total references: TODO


Result: TODO