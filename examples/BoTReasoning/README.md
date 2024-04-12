
# Boosting of Thought (BoT) algorithm

The structure of our proposed Boosting of Thoughts (BoT) under `examples/BoTReasoning/` is 

    .
    ├── BoTReasoning                     # Main Reasoning Pipeline
    ├── reasoner                         # Perform Reasoning for each sample
    ├── thought_model                    # Model for the Thought Operation
    ├── thought_prompter                 # Prompter for the Thought Operation
    ├── commenter                        # Model for the Comment Operation
    ├── comment_prompter                 # Prompter for the Comment Operation
    ├── aggregator                       # Aggregator for the solution ensemble


## Boosting of Thoughts (BoT) algorithm

Following the commends below to run the BoT algorithm on different datasets:

```console
python examples/BoTReasoning/BoTReasoning.py -c configs/GameOf24/GPT4/BoT_ZeroshotCoT.yml -b ICLR

python examples/BoTReasoning/BoTReasoning.py -c configs/GSM8K/GPT4/BoT_ZeroshotCoT.yml -b ICLR

python examples/BoTReasoning/BoTReasoning.py -c configs/SVAMP/GPT4/BoT_ZeroshotCoT.yml -b ICLR

python examples/BoTReasoning/BoTReasoning.py -c configs/AQUA/GPT4/BoT_ZeroshotCoT.yml -b ICLR

python examples/BoTReasoning/BoTReasoning.py -c configs/MATH/GPT4/BoT_ZeroshotCoT.yml -b ICLR

python examples/BoTReasoning/BoTReasoning.py -c configs/TheoremQA/GPT4/BoT_ZeroshotCoT.yml -b ICLR
```
