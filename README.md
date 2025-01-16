# Overview
`llmpebase` is a unified platform designed to support experiments with various reasoning methods for large language models (LLMs) in problem-solving. This codebase is designed to be _easy to use_ and ensures fair comparisons across methods. It provides modular components that facilitate the implementation of new reasoning algorithms and their application to diverse datasets across various tasks for comprehensive evaluation.

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
- BoT prompting
- TR prompting

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

## References

[1]. Chen, Sijia and Li, Baochun and Niu, Di, Boosting of thoughts: Trial-and-error problem solving with large language models, ICLR24. See `examples/BoTReasoning`

[2]. Chen, Sijia and Li, Baochun, Toward Adaptive Reasoning in Large Language Models with Thought Rollback, ICML24. See `examples/ThoughtRollback`
