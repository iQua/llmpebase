"""
An interface to evaluate the performance by measuring the similarity between results and 
groundtruths.
"""


from llmpebase.evaluator.re_evaluation import GSM8KEvaluator, GSM8KLLMEvaluator


basic_evaluators = {"GSM8K": GSM8KEvaluator}


llm_evaluators = {"GSM8K": GSM8KLLMEvaluator}


def get(data_name, style="basic", **kwargs):
    """Get the evaluators for the specific dataset."""

    assert style in ["basic", "llm"]

    evaluators = basic_evaluators if style == "basic" else llm_evaluators

    if data_name in evaluators:
        return evaluators[data_name]

    else:
        raise NotImplementedError(
            f"{style} evaluator is not implemented for {data_name}"
        )
