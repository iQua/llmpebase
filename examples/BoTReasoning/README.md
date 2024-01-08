
# Boosting of Thought (BoT) algorithm

The structure of our proposed Boosting of Thoughts (BoT) under `examples/BoostingOfThought/` is 

    .
    ├── BoT_reasoner                     # Reasoner relying on experiences
    ├── BoT_commenter                    # Perform Thought Chain Analysis
    ├── BoT_aggregator                   # Perform Thought Structures Aggregation
    ├── BoT_core                         # The core reasoning process with Iterative Refinement of BoT
    ├── BoT                              # Main function to run experiments

## Boosting of Thoughts (BoT) algorithm

To apply the proposed BoT algorithm on the Game of 24, which is the most difficult task mentioned in the submission, one only needs to run the following command.

```console
python examples/BoostingOfThought/BoT.py -c configs/GameOf24/BoT_chatgpt.yml -b ICLR
python examples/BoostingOfThought/BoT.py -c configs/GameOf24/BoTSingle_chatgpt.yml -b ICLR
```

The configuration file `BoTSingle_chatgpt.yml` is used to perform the ablation study of BoT by using one single tree and a single reasoning process. This BoT variant can be compared with the tree of thoughts [1], which performs the reasoning by building thoughts in a tree structure.


[1]. Tree of Thoughts: Deliberate Problem Solving with Large Language Models, 2023.