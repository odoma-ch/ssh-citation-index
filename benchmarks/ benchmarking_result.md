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
| Mistral-Small-3.2-24B-Instruct-2506     | 0.7923 | 0.7985 | 0.7928   | 0.7770   | 20            | 1421.17     |   0.7699    |   0.7747     |       0.8689           |
| Claude 4 Sonnet                         |     |     |       |       | -             |          |       |        |                  |




#### reference extraction and parsing :

| LLM                                  | P    | R    | micro F1   | Macro F1   | # Failed Docs | runtime (s) | Title F1 | Author F1 | publication year F1 |
|------------------------------------------|------|------|------|--------------|---------------|-------------|----------|----------|--------------------|
| Deepseek v3                              | 0.8896 | 0.8185 | 0.8213 | 0.7834 | 7 | 1658.21 s      |   0.8177    |   0.8195    |         0.8289        |
| Gemma 3 27b                              | 0.8557 | 0.7193 | 0.7455 | 0.6755 |  9 | 3503.94 s      |   0.7373    |   0.7498    |         0.7473        |
| Mistral-Small-3.2-24B-Instruct-2506      | 0.6134 | 0.6389 | 0.6140 | 0.5860 |  35  | 5817.9s     |   0.6170    |   0.5993    |         0.6307        |
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
| Deepseek-chat                   | 0.7044 | 0.4460 | 0.4728   | 0.4413   | 26               | 929.94      | 0.4516   | 0.4726    | 0.4894             |
| Gemma-3-27b-it                | 0.9040 | 0.4606 | 0.5446   | 0.4523   | 0                | 2004.67     | 0.5115   | 0.5495    | 0.5545             |
| Mistral-Small-3.1-24B          | 0.8578 | 0.8645 | 0.8513   | 0.8397   | 0                | 4046.45     | 0.8178   | 0.8424    | 0.8904             |


**Different methods of reference extraction_and_parsing:**

- Method 1: One single LLM call on the full text
- Method 2: Two separate LLM calls, one for the reference extraction and one for the reference parsing
- Method 3: Heuristic section detection without LLM, then one single LLM call for the reference parsing
- Method 4: Page-wise extraction and reference parsing with LLM  
- Method 5: Page-wise extraction with LLM, then one single LLM call for the reference parsing *still working on it*

#### Methods comparison for extraction and parsing:

| Model                        | Method | P      | R      | micro F1 | macro F1 | # Parsing Errors | runtime (s) | Title F1 | Author F1 | Publication Date F1 |
|-----------------------------------------|--------|--------|--------|----------|----------|------------------|-------------|----------|-----------|---------------------|
| Mistral-Small-3.1-24B          | 1      | 0.8578 | 0.8645 | 0.8513   | 0.8397   | 0                | 4046.45     | 0.8178   | 0.8424    | 0.8904             |
| Mistral-Small-3.1-24B          | 2      | 0.8996 | 0.9113 | 0.9044   | 0.8936   | 0                | 3841.06     | 0.8839   | 0.8986    | 0.9410             |
| Mistral-Small-3.1-24B          | 3      | 0.4541 | 0.3203 | 0.3295   | 0.3174   | 0                | 1988.72     | 0.3096   | 0.3266    | 0.3532             |
| Mistral-Small-3.1-24B          | 4      | 0.8485 | 0.8733 | 0.8524   | 0.8425   | 1                | 19103.29    | 0.8241   | 0.8438    | 0.8979             |


# Linkedbook dataset

Here you go—updated exactly as requested.

### Run summary 

Sorted by **Overall micro-F1** (descending) within each mode.

| Mode    | Model                                                           | LLM Duration (s) | Overall P | Overall R | Overall micro-F1 | Overall macro-F1 | Focused P | Focused R | Focused micro-F1 | Focused macro-F1 |
| ------- | --------------------------------------------------------------- | ---------------: | --------: | --------: | ---------------: | ---------------: | --------: | --------: | ---------------: | ---------------: |
| single  | deepseek-chat <i>(fewshot)</i>                                  |           386.91 |    0.6730 |    0.8861 |           0.7650 |           0.7605 |    0.8656 |    0.8717 |           0.8686 |           0.8572 |
| single  | google/gemma-3-27b-it <i>(fewshot)</i>                          |            207.8 |    0.6613 |    0.8944 |           0.7604 |           0.7650 |    0.8560 |    0.8822 |           0.8689 |           0.8667 |
| single  | mistralai/Mistral-Small-3.1-24B-Instruct-2503 <i>(fewshot)</i>  |           128.41 |    0.6539 |    0.8823 |           0.7512 |           0.7471 |    0.8485 |    0.8677 |           0.8580 |           0.8450 |
| single  | mistralai/Mistral-Small-3.1-24B-Instruct-2503 <i>(zeroshot)</i> |           143.93 |    0.6668 |    0.8246 |           0.7374 |           0.6918 |    0.8462 |    0.8052 |           0.8252 |           0.7723 |
| grouped | deepseek-chat <i>(fewshot)</i>                                  |           305.61 |    0.6906 |    0.7572 |           0.7224 |           0.7161 |    0.8714 |    0.7476 |           0.8047 |           0.7999 |
| grouped | google/gemma-3-27b-it <i>(fewshot)</i>                          |           195.15 |    0.6973 |    0.4445 |           0.5429 |           0.5210 |    0.8585 |    0.4440 |           0.5853 |           0.5812 |
| grouped | mistralai/Mistral-Small-3.1-24B-Instruct-2503 <i>(fewshot)</i>  |           224.56 |    0.6697 |    0.8932 |           0.7655 |           0.7697 |    0.8582 |    0.8753 |           0.8666 |           0.8615 |

### Per-field micro-F1 (Focused subset)

| Run (model<i>(prompt)</i> • mode)                             | authors | full\_title | publication\_date |
| ------------------------------------------------------------- | ------: | ----------: | ----------------: |
| Mistral-24B <i>(fewshot)</i> • grouped |  0.8042 |      0.8845 |            0.9074 |
| Gemma-27B <i>(fewshot)</i> • single    |  0.8063 |      0.8895 |            0.9082 |
| Mistral-24B <i>(fewshot)</i> • single  |  0.7982 |      0.8724 |            0.9015 |
| Gemma-27B <i>(fewshot)</i> • grouped   |  0.5539 |      0.5916 |            0.6078 |
| DeepSeek <i>(fewshot)</i> • single     |  0.8048 |      0.8848 |            0.9117 |
| DeepSeek <i>(fewshot)</i> • grouped    |  0.7640 |      0.8147 |            0.8325 |
| Mistral-24B <i>(zeroshot)</i> • single   |  0.7675 |      0.8371 |            0.8697 |

### Per-language F1 (overall) — only for runs that reported it

| Run (model<i>(prompt)</i> • mode)                            |     DE |     EN |     ES |     FR |     IT |     NL |     PT |
| ------------------------------------------------------------ | -----: | -----: | -----: | -----: | -----: | -----: | -----: |
| Gemma-27B <i>(fewshot)</i> • single   | 0.8177 | 0.7322 | 0.7608 | 0.7858 | 0.7631 | 0.8889 | 0.7360 |
| Mistral-24B <i>(fewshot)</i> • single | 0.8008 | 0.7299 | 0.7759 | 0.7835 | 0.7525 | 0.8889 | 0.6305 |
| DeepSeek <i>(fewshot)</i> • single    | 0.8110 | 0.7417 | 0.7894 | 0.7803 | 0.7679 | 0.8117 | 0.7003 |
| Mistral-24B <i>(zeroshot)</i> • single | 0.7697 | 0.7377 | 0.7166 | 0.7766 | 0.7342 | 0.7500 | 0.5287 |
