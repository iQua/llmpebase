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

    solution_flag: str = "Number in New Set:"

    instruction: str = (
        "Within the latest step, 'Two Numbers' are selected from the 'Current Set' to perform +,-,*,/, i.e. Operation, to obtain a 'New Number'. Then, 'Current Set' removes these 'Two Numbers' and thus gets 'Remaining Numbers'. Then, 'New Set' will be the combination of 'New Number' and 'Remaining Numbers'. Report the solution when the 'New Set' contains only one number."
    )

    step_format: str = (
        "Step idx: Current Set= ,\tTwo Numbers= ,\tOperation= ,\tNew Number ,\tRemaining Numbers= ,\tNew Set= "
    )

    question_prompt_head: str = (
        f"""In the game of 24, you are given four numbers, and each number can be used only once. The goal is to use basic arithmetic operations (+, -, *, /) to combine these numbers and obtain 24 in the New Set.\nRule: {instruction}\nStep Format: {step_format}.\n"""
    )

    question_format = base.BasicPromptFormat(
        head=question_prompt_head,
        content="Four given numbers are: {}",
        notice=" ",
        tail="\n",
    )
