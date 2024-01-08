# Overview
`llmpebase` is a unified platform to support experiments for prompt engineering in large language models (LLMs). This codebase is designed to be _easy to use_ and to ensure a fair comparison. With the components provided by this codebase, one can easily implement a new prompt engineering algorithm and apply it to a task for evaluation.

## Code structure
The structure of `llmpebase` is 

    .
    ├── configs                         # Configuration files to be used
    ├── examples                        # Implemented examples
    ├── llmpebase                       # The source code of `llmpebase`
    └──── datasets                       # Datasets
    └──── models                         # LLMs, prompting, and thought structures
    └──── exactor                        # To extract the result from the output 
    └──── evaluator                      # To evaluate by comparing ground truth and the result

# Implemented Components

## Task and dataset
Mathematical problems:

- GSM8K
- SVAMP
- AQUA-RAT
- MATH
- TheoremQA
- Game of 24

Multi-task Reasoning:

- MMLU
- BBH

Commonsense reasoning:

- CSQA (CommonsenseQA)


## Large Language Models (LLMs)

- GPT
- Llama
- Llama2
- Claude
- Falcon

## Prompting

- Few-shot prompting
- Chain-of-Thought (CoT) prompting
- Zero-shot prompting
- Thought prompting

## Thought structures

- Chain 
- Tree
- Graph

## Extractor

- LLM-based extractor
- Regex-based extractor

## Evaluator

- Regex-based evaluator


# How to use 
Anyone can run `examples/` of `llmpebase` by executing the following three steps: 

0. (Optional). To use ChatGPT API, one needs to have the _OPENAI_API_KEY_, _OPENAI_ORGAN_KEY_, and _ANTHROPIC_KEY_ and set them in the file `.env` under the root directory.

1. Download the code from GitHub. Install `llmpebase` by running 
    ```console
    $ pip install .
    ```

2. Run the example by running 
    ```console
    $ python examples/ChainOfThought/ChainOfThought.py -c configs/GSM8K/Standard_chatgpt.yml -b LLMPEBASE
    $ python examples/ChainOfThought/ChainOfThought.py -c configs/GSM8K/CoT_chatgpt.yml -b LLMPEBASE
    $ python examples/ChainOfThought/ChainOfThought.py -c configs/GSM8K/ZeroCoT_chatgpt.yml -b LLMPEBASE
    ```
    See documents under `examples/` for more details.