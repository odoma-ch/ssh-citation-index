## Targeting model
- Deepseek v3 (API)
- Claude 4 Sonnet (API)
- Gemma 3 27b (PSNC VM)
- mistralai/Mistral-Small-3.2-24B-Instruct-2506 (PSNC VM)
- numind/NuExtract-2.0-8B (PSNC VM)
- 


### EXCITE dataset

Total documents: 351
Total pages: 8041
Total references: 10171


#### reference extraction :

Here the failure reason mainly because the pdf is too large for the context window.

| LLM                                      | P    | R    | F1   | Similarity   | # Failed Docs | runtime (s) |
|------------------------------------------|------|------|------|--------------|---------------|-------------|
| Deepseek v3                              | 0.7272 | 0.7375 | 0.7229 | 0.7848 | 13          |          |
| Gemma 3 27b                              | 0.5474 | 0.6726 | 0.5893 | 0.8072 | 13 | 2549.77s        |
| Mistral-Small-3.2-24B-Instruct-2506      | 0.7815 | 0.9319 | 0.8336 | 0.9629 | 13 | 3215.79s   |
| Claude 4 Sonnet                          |   |   |   |           |            |          |



#### reference parsing :

| LLM                                     |   P    |   R    | micro F1 | macro F1 | # Failed Docs | runtime (s) | Title F1 | Author F1 | Publication Year F1 |
|-----------------------------------------|--------|--------|----------|----------|---------------|-------------|----------|-----------|---------------------|
| Deepseek v3 (API)                       | 0.8207 | 0.7789 | 0.7817   | 0.7594   | 7             | 1612.73     |  0.7552     |     0.7700   |       0.8468           |
| Gemma 3 27b                             | 0.8102 | 0.8094 | 0.8054   | 0.7879   | 10            | 2245.30     |    0.7693  |     0.7793    |      0.8993            |
| Mistral-Small-3.2-24B-Instruct-2506     | 0.7923 | 0.7985 | 0.7928   | 0.7770   | 20            | 1421.17     |       |        |                  |
| Claude 4 Sonnet                         |     |     |       |       | -             |          |       |        |                  |




#### reference extraction and parsing :

| LLM                                  | P    | R    | micro F1   | Macro F1   | # Failed Docs | runtime (s) | Title F1 | Author F1 | publication year F1 |
|------------------------------------------|------|------|------|--------------|---------------|-------------|----------|----------|--------------------|
| Deepseek v3                              | 0.8543 | 0.6118 | 0.713 | 0.6371 | 24 |          |       |       |                 |
| Gemma 3 27b                              | 0.6514 | 0.5687 | 0.5863 | 0.5393 |  9 | 3503.94 s      |       |                 |
| Mistral-Small-3.2-24B-Instruct-2506      | 0.5077 | 0.5388 | 0.5165 | 0.4965 |   | 5817.9s     |       |       |                 |
| Claude 4 Sonnet                          |   |   |   |           |            | -           |       |       |                 |



## CEXgoldstandard dataset

Total documents:  107
Total references:   5008


Result:  

#### reference extraction :

| LLM                                      | Avg GT Count | Avg Pred Count | Avg Difference | Avg % Difference | Over/Under/Exact | runtime (s) |
|------------------------------------------|--------------|----------------|----------------|------------------|------------------|-------------|
| Deepseek-chat                    | 46.8         | 43.04          | -3.77          | -4.91%           | 4/29/74          | 617.00      |
| Gemma-3-27b-it                   | 46.8         | 32.78          | -14.03         | -20.57%          | 18/75/14         | 947.60      |
| Mistral-Small-3.1-24B            | 46.8         | 47.26          | +0.46          | +0.81%           | 4/7/96           | 1712.73     |

*Note: Over/Under/Exact represents documents with over-extracted/under-extracted/exact match reference counts.*



#### reference parsing :

| LLM                                     | P      | R      | micro F1 | macro F1 | # Parsing Errors | runtime (s) | Title F1 | Author F1 | Publication Date F1 |
|-----------------------------------------|--------|--------|----------|----------|------------------|-------------|----------|-----------|---------------------|
| Deepseek-chat                  | 0.7775 | 0.4789 | 0.5124   | 0.4744   | 23               | 872.21      | 0.5075   | 0.5098    | 0.5217             |
| Gemma-3-27b-it                | 0.9907 | 0.9146 | 0.9325   | 0.9092   | 0                | 1882.01     | 0.9234   | 0.9332    | 0.9374             |
| Mistral-Small-3.1-24B          | 0.9800 | 0.9760 | 0.9779   | 0.9700   | 0                | 1783.32     | 0.9787   | 0.9748    | 0.9893             |



#### reference extraction and parsing (Method 1: Single LLM call) :

| LLM                          | P      | R      | micro F1 | macro F1 | # Parsing Errors | runtime (s) | Title F1 | Author F1 | Publication Date F1 |
|-----------------------------------------|--------|--------|----------|----------|------------------|-------------|----------|-----------|---------------------|
| Deepseek-chat                  | 0.7584 | 0.3303 | 0.3659   | 0.3257   | 18               | 604.65      | 0.3405   | 0.3681    | 0.3798             |
| Gemma-3-27b-it                | 0.9040 | 0.4606 | 0.5446   | 0.4523   | 0                | 2004.67     | 0.5115   | 0.5495    | 0.5545             |
| Deepseek-chat                   | 0.7292 | 0.3805 | 0.4123   | 0.3771   | 24               | 929.94      | 0.3924   | 0.4123    | 0.4302             |
| Mistral-Small-3.1-24B          | 0.8578 | 0.8645 | 0.8513   | 0.8397   | 0                | 4046.45     | 0.8178   | 0.8424    | 0.8904             |


**Different methods of reference extraction_and_parsing:**

Method 1: One single LLM call on the full text
Method 2: Two separate LLM calls, one for the reference extraction and one for the reference parsing
Method 3: Heuristic section detection without LLM, then one single LLM call for the reference parsing
Method 4: Page-wise extraction and reference parsing with LLM  *still working on it*
Method 5: Page-wise extraction with LLM, then one single LLM call for the reference parsing

#### Methods comparison for extraction and parsing:

| Model                        | Method | P      | R      | micro F1 | macro F1 | # Parsing Errors | runtime (s) | Title F1 | Author F1 | Publication Date F1 |
|-----------------------------------------|--------|--------|--------|----------|----------|------------------|-------------|----------|-----------|---------------------|
| Mistral-Small-3.1-24B          | 1      | 0.8578 | 0.8645 | 0.8513   | 0.8397   | 0                | 4046.45     | 0.8178   | 0.8424    | 0.8904             |
| Mistral-Small-3.1-24B          | 2      | 0.8996 | 0.9113 | 0.9044   | 0.8936   | 0                | 3841.06     | 0.8839   | 0.8986    | 0.9410             |
| Mistral-Small-3.1-24B          | 3      | 0.4541 | 0.3203 | 0.3295   | 0.3174   | 0                | 1988.72     | 0.3096   | 0.3266    | 0.3532             |
| Mistral-Small-3.1-24B          | 4      | 0.8485 | 0.8733 | 0.8524   | 0.8425   | 1                | 19103.29    | 0.8241   | 0.8438    | 0.8979             |

