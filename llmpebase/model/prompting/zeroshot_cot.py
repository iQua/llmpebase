"""
Implementations of Zero-shot CoT prompting
"""

from llmpebase.model.prompting import base


class TheoremQAZeroShotCoTPrompting(base.BaseZeroShotCoTPrompting):
    """The zeroshot CoT prompt of TheoremQA."""

    solution_flag: str = "The final solution is"


class MMLUZeroShotCoTPrompting(base.BaseZeroShotCoTPrompting):
    """The zeroshot CoT prompt of MMLU."""

    solution_flag: str = "The final choice is"


class CSQAZeroShotCoTPrompting(base.BaseZeroShotCoTPrompting):
    """The zeroshot CoT prompt of CommonsenseQA."""

    solution_flag: str = "The final choice is"


class AQUAZeroShotCoTPrompting(base.BaseZeroShotCoTPrompting):
    """The zeroshot CoT prompt of AQUA-RAT."""

    solution_flag: str = "The final choice is"


class GameOf24ZeroShotCoTPrompting(base.BaseZeroShotCoTPrompting):
    """The zeroshot prompt of GameOf24."""

    solution_flag: str = ""

    instruction: str = (
        "Within the latest step, 'Two Numbers' are selected from the 'Current Set' to perform + or - or * or /, i.e. Operation, to obtain a 'New Number'. Then, 'Current Set' removes these 'Two Numbers' and thus gets 'Remaining Numbers'. Then, 'New Set' will be the combination of 'New Number' and 'Remaining Numbers'."
    )

    step_format: str = (
        "Step idx: Current Set= ,\tTwo Numbers= ,\tOperation= ,\tNew Number ,\tRemaining Numbers= ,\tNew Set= "
    )

    question_prompt_head: str = (
        f"""You are given four numbers and the Rule and Step Format should be followed to address the Game of 24 problem.\nRule: {instruction}\nStep Format: {step_format}.\n"""
    )

    question_format = base.BasicPromptFormat(
        head=question_prompt_head,
        content="Four given numbers are: {}",
        notice=" ",
        tail="\n",
    )
