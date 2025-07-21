## Targeting model
- Deepseek v3 (API)
- Claude 4 Sonnet (API)
- Gemma 3 27b (PSNC VM)
- mistralai/Mistral-Small-3.2-24B-Instruct-2506 (PSNC VM)
- 


### EXCITE dataset

Total documents: 351
Total pages: 8041
Total references: 10171


#### reference extraction :

Here the failure reason mainly because the pdf is too large for the context window.

| LLM                                      | P    | R    | F1   | Similarity   | # Failed Docs | runtime (s) |
|------------------------------------------|------|------|------|--------------|---------------|-------------|
| Deepseek v3                              | 0.7272 | 0.7375 | 0.7229 | 0.7848 | 13          | TODO        |
| Claude 4 Sonnet                          | TODO | TODO | TODO | TODO         | TODO          | TODO        |
| Gemma 3 27b                              | TODO | TODO | TODO | TODO         | TODO          | TODO        |
| Mistral-Small-3.2-24B-Instruct-2506      | TODO | TODO | TODO | TODO         | TODO          | TODO        |


#### reference parsing :

| LLM                                     |   P    |   R    | micro F1 | macro F1 | # Failed Docs | runtime (s) | Title F1 | Author F1 | Publication Year F1 |
|-----------------------------------------|--------|--------|----------|----------|---------------|-------------|----------|-----------|---------------------|
| Deepseek v3 (API)                       | 0.8207 | 0.7789 | 0.7817   | 0.7594   | 6             | 1612.73     | TODO     | TODO      | TODO                |
| Claude 4 Sonnet                         | TODO   | TODO   | TODO     | TODO     | -             | TODO        | TODO     | TODO      | TODO                |
| Gemma 3 27b                             | 0.8102 | 0.8094 | 0.8054   | 0.7879   | 10            | 2245.30     | TODO     | TODO      | TODO                |
| Mistral-Small-3.2-24B-Instruct-2506     | 0.7923 | 0.7985 | 0.7928   | 0.7770   | 20            | 1421.17     | TODO     | TODO      | TODO                |



#### reference extraction and parsing :

| LLM                                  | P    | R    | micro F1   | Macro F1   | # Failed Docs | runtime (s) | Title F1 | Author F1 | publication year F1 |
|------------------------------------------|------|------|------|--------------|---------------|-------------|----------|----------|--------------------|
| Deepseek v3                              | 0.8543 | 0.6118 | 0.713 | 0.6371 | 24 | TODO        | TODO     | TODO     | TODO               |
| Claude 4 Sonnet                          | TODO | TODO | TODO | TODO         | TODO          | -           | TODO     | TODO     | TODO               |
| Gemma 3 27b                              | 0.6514 | 0.5687 | 0.5863 | 0.5393 |  9 | 3503.94 s      | TODO     | TODO               |
| Mistral-Small-3.2-24B-Instruct-2506      | 0.5077 | 0.5388 | 0.5165 | 0.4965 | TODO | 5817.9s     | TODO     | TODO     | TODO               |



## CEXgoldstandard dataset

Total documents: TODO
Total references: TODO


Result: TODO