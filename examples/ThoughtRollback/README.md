# Overview
After installing the `llmpebase` package, you can run our Thought Rollback with different LLMs under different datasets.

Here are some examples:

#### GSM8K
```console
python examples/ChainOfThought/ChainOfThought.py -c configs/GSM8K/GPT4/Fewshot_chatgpt.yml -b LLMPE
python examples/ChainOfThought/ChainOfThought.py -c configs/GSM8K/GPT4/CoT_chatgpt.yml -b LLMPE
python examples/ChainOfThought/ChainOfThought.py -c configs/GSM8K/GPT4/ZeroshotCoT_chatgpt.yml -b LLMPE

python examples/ChainReasoning/ChainReasoning.py -c configs/GSM8K/GPT4/ChainReasoning_ZeroshotCoT_chatgpt.yml -b LLMPE
python examples/TreeReasoning/TreeReasoning.py -c configs/GSM8K/GPT4/TreeLWGReasoning_ZeroshotCoT_chatgpt.yml -b LLMPE
python examples/ThoughtRollback/ThoughtRollback.py -c configs/GSM8K/GPT4/TRReasoning_ZeroshotCoT_chatgpt.yml -b LLMPE
```


#### SVAMP
```console
python examples/ChainOfThought/ChainOfThought.py -c configs/SVAMP/GPT4/Fewshot_chatgpt.yml -b LLMPE
python examples/ChainOfThought/ChainOfThought.py -c configs/SVAMP/GPT4/CoT_chatgpt.yml -b LLMPE
python examples/ChainOfThought/ChainOfThought.py -c configs/SVAMP/GPT4/ZeroshotCoT_chatgpt.yml -b LLMPE

python examples/ChainReasoning/ChainReasoning.py -c configs/SVAMP/GPT4/ChainReasoning_ZeroshotCoT_chatgpt.yml -b LLMPE
python examples/TreeReasoning/TreeReasoning.py -c configs/SVAMP/GPT4/TreeLWGReasoning_ZeroshotCoT_chatgpt.yml -b LLMPE
python examples/ThoughtRollback/ThoughtRollback.py -c configs/SVAMP/GPT4/TRReasoning_ZeroshotCoT_chatgpt.yml -b LLMPE
```

#### AQUA
```console
python examples/ChainOfThought/ChainOfThought.py -c configs/AQUA/GPT4/Fewshot_chatgpt.yml -b LLMPE
python examples/ChainOfThought/ChainOfThought.py -c configs/AQUA/GPT4/CoT_chatgpt.yml -b LLMPE
python examples/ChainOfThought/ChainOfThought.py -c configs/AQUA/GPT4/ZeroshotCoT_chatgpt.yml -b LLMPE

python examples/ChainReasoning/ChainReasoning.py -c configs/AQUA/GPT4/ChainReasoning_ZeroshotCoT_chatgpt.yml -b LLMPE
python examples/TreeReasoning/TreeReasoning.py -c configs/AQUA/GPT4/TreeLWGReasoning_ZeroshotCoT_chatgpt.yml -b LLMPE
python examples/ThoughtRollback/ThoughtRollback.py -c configs/AQUA/GPT4/TRReasoning_ZeroshotCoT_chatgpt.yml -b LLMPE
```

	



#### MATH
GPT-3.5-turbo:
```console
python examples/ChainOfThought/ChainOfThought.py -c configs/MATH/Fewshot_chatgpt.yml -b LLMPE
python examples/ChainOfThought/ChainOfThought.py -c configs/MATH/CoT_chatgpt.yml -b LLMPE
python examples/ChainOfThought/ChainOfThought.py -c configs/MATH/ZeroshotCoT_chatgpt.yml -b LLMPE
python examples/ChainOfThought/ChainOfThought.py -c configs/MATH/Zeroshot_chatgpt.yml -b LLMPE
```

GPT-4:

```console
python examples/ChainOfThought/ChainOfThought.py -c configs/MATH/GPT4/Fewshot_chatgpt.yml -b LLMPE
python examples/ChainOfThought/ChainOfThought.py -c configs/MATH/GPT4/CoT_chatgpt.yml -b LLMPE
python examples/ChainOfThought/ChainOfThought.py -c configs/MATH/GPT4/ZeroshotCoT_chatgpt.yml -b LLMPE
python examples/ChainOfThought/ChainOfThought.py -c configs/MATH/GPT4/Zeroshot_chatgpt.yml -b LLMPE

python examples/ChainReasoning/ChainReasoning.py -c configs/MATH/GPT4/ChainReasoning_ZeroshotCoT_chatgpt.yml -b LLMPE
python examples/TreeReasoning/TreeReasoning.py -c configs/MATH/GPT4/TreeLWGReasoning_ZeroshotCoT_chatgpt.yml -b LLMPE
python examples/ThoughtRollback/ThoughtRollback.py -c configs/MATH/GPT4/TRReasoning_ZeroshotCoT_chatgpt.yml -b LLMPE

```

#### MMLU

```console
python examples/ChainOfThought/ChainOfThought.py -c configs/MMLU/GPT4/Fewshot_chatgpt.yml -b LLMPE
python examples/ChainOfThought/ChainOfThought.py -c configs/MMLU/GPT4/ZeroshotCoT_chatgpt.yml -b LLMPE

python examples/ChainReasoning/ChainReasoning.py -c configs/MMLU/GPT4/ChainReasoning_ZeroshotCoT_chatgpt.yml -b LLMPE
python examples/TreeReasoning/TreeReasoning.py -c configs/MMLU/GPT4/TreeLWGReasoning_ZeroshotCoT_chatgpt.yml -b LLMPE
python examples/ThoughtRollback/ThoughtRollback.py -c configs/MMLU/GPT4/TRReasoning_ZeroshotCoT_chatgpt.yml -b LLMPE

```

#### TheoremQA

```console
python examples/ChainOfThought/ChainOfThought.py -c configs/TheoremQA/GPT4/Fewshot_chatgpt.yml -b LLMPE
python examples/ChainOfThought/ChainOfThought.py -c configs/TheoremQA/GPT4/CoT_chatgpt.yml -b LLMPE
python examples/ChainOfThought/ChainOfThought.py -c configs/TheoremQA/GPT4/ZeroshotCoT_chatgpt.yml -b LLMPE

python examples/ChainReasoning/ChainReasoning.py -c configs/TheoremQA/GPT4/ChainReasoning_ZeroshotCoT_chatgpt.yml -b LLMPE
python examples/TreeReasoning/TreeReasoning.py -c configs/TheoremQA/GPT4/TreeLWGReasoning_ZeroshotCoT_chatgpt.yml -b LLMPE
python examples/ThoughtRollback/ThoughtRollback.py -c configs/TheoremQA/GPT4/TRReasoning_ZeroshotCoT_chatgpt.yml -b LLMPE
```




