# Overview
We built the codebase `llmpebase` as a unified platform to perform prompt engineering in large language models (LLMs). The currently released code supports the experiments of ChatGPT and LlamaV2 models on the Game of 24, GSM8K, and MMLU datasets with the corresponding task prompts. `llmpebase` is designed to support fairness comparison and reproducibility. More importantly, `llmpebase` is _easy to use_. Therefore, it was used to perform all experiments and analyses in the ICLR submission. 

## Code structure
The structure of `llmpebase` is 

    .
    ├── configs                         # Configuration files used in experiments
    ├── examples                        # Implementation of BoT and baseline 
    ├── llmpebase                       # The source code of llmpebase
    └──── datasets                       # Datasets used in experiments
    └──── models                         # Large Language Models (LLMs) and Prompting design of different tasks
    └──── utils                          # Recorder used in evaluation

The structure of our proposed Boosting of Thoughts (BoT) under `examples/BoostingOfThought/` is 

    .
    ├── BoT_reasoner                     # Reasoner relying on experiences
    ├── BoT_commenter                    # Perform Thought Chain Analysis
    ├── BoT_aggregator                   # Perform Thought Structures Aggregation
    ├── BoT_core                         # The core reasoning process with Iterative Refinement of BoT
    ├── BoT                              # Main function to run experiments

Finally, the tree thought structures mentioned in our submission are implemented in `llmpebase/models/prompting/tree_thoughts.py`.


# How to use 
Anyone can run `examples/` of `llmpebase` by executing the following three steps: 

0. (Optional). To use ChatGPT API, one needs to have the _OPENAI_API_KEY_, _OPENAI_ORGAN_KEY_, and _ANTHROPIC_KEY_ and set them in the file `.env` under the root directory.

1. Install `llmpebase` by running 
    ```console
    $ pip install .
    ```

2. Visit `examples/` to decide to run which algorithm, such as performing Boosting of Thoughts (BoT) on the Game of 24 dataset:
    ```console
    $ python examples/BoostingOfThought/BoT.py -c configs/GameOf24/BoT_chatgpt.yml -b ICLR
    ```

3. Collect results from the folder `ICLR/results/`, which contains
    - `llm_records/records.json` recording the input prompt, response, extract answer, whether the result is correct or not for each question.
    - `llm_records/samples.json` recording loaded samples including the answer and ground truth result for each question.

# Reproduce experiments of the submission 

Limited by space, we provide two streams of experiments, including the baseline Prompt engineering algorithms and the proposed Boosting of thoughts (BoT) algorithm. 

## Baseline Prompt engineering algorithms

There are three baseline algorithms, including Few-shot prompting, Chain-of-Thought (CoT) prompting, and Zero Shot Chain-of-Thought. To apply these algorithms on the Game of 24, GSM8K, and MMLU datasets, one may need to perform the following comments.

1. Perform these algorithms with ChatGPT on the GSM8K dataset by running 
    ```console
    $ python examples/ChainOfThought/ChainOfThought.py -c configs/GSM8K/Standard_chatgpt.yml -b ICLR
    $ python examples/ChainOfThought/ChainOfThought.py -c configs/GSM8K/CoT_chatgpt.yml -b ICLR
    $ python examples/ChainOfThought/ChainOfThought.py -c configs/GSM8K/ZeroCoT_chatgpt.yml -b ICLR
    ```

2. Perform these algorithms with ChatGPT on the MMLU dataset by running
    ```console
    $ python examples/ChainOfThought/ChainOfThought.py -c configs/MMLU/Standard_chatgpt.yml -b ICLR 
    $ python examples/ChainOfThought/ChainOfThought.py -c configs/MMLU/CoT_chatgpt.yml -b ICLR 
    $ python examples/ChainOfThought/ChainOfThought.py -c configs/MMLU/ZeroCoT_chatgpt.yml -b ICLR 
    ```

3. Perform these algorithms with ChatGPT on the Game of 24 dataset by running
    ```console
    $ python examples/ChainOfThought/ChainOfThought.py -c configs/GameOf24/Standard_chatgpt.yml -b ICLR 
    $ python examples/ChainOfThought/ChainOfThought.py -c configs/GameOf24/CoT_chatgpt.yml -b ICLR 
    $ python examples/ChainOfThought/ChainOfThought.py -c configs/GameOf24/ZeroCoT_chatgpt.yml -b ICLR 
    ```


## Boosting of Thoughts (BoT) algorithm

To apply the proposed BoT algorithm on the Game of 24, which is the most difficult task mentioned in the submission, one only needs to run the following command.

```console
python examples/BoostingOfThought/BoT.py -c configs/GameOf24/BoT_chatgpt.yml -b ICLR
python examples/BoostingOfThought/BoT.py -c configs/GameOf24/BoTSingle_chatgpt.yml -b ICLR
```

The configuration file `BoTSingle_chatgpt.yml` is used to perform the ablation study of BoT by using one single tree and a single reasoning process. This BoT variant can be compared with the tree of thoughts [1], which performs the reasoning by building thoughts in a tree structure.

[1]. Tree of Thoughts: Deliberate Problem Solving with Large Language Models, 2023.

