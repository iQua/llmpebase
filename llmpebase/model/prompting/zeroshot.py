"""
Implementations of Zero-shot prompting
"""

from llmpebase.model.prompting import base
from llmpebase.model.prompting import zeroshot_cot


class TheoremQAZeroShotPrompting(base.BaseZeroShotPrompting):
    """The zeroshot prompt of TheoremQA."""

    solution_flag: str = "The final solution is"


class MMLUZeroShotPrompting(base.BaseZeroShotPrompting):
    """The zeroshot prompt of MMLU."""

    solution_flag: str = "The final choice is"


class CSQAZeroShotPrompting(base.BaseZeroShotPrompting):
    """The zeroshot prompt of CommonsenseQA."""

    solution_flag: str = "The final choice is"


class AQUAZeroShotPrompting(base.BaseZeroShotPrompting):
    """The zeroshot prompt of AQUA-RAT."""

    solution_flag: str = "The final choice is"


class GameOf24ZeroShotPrompting(zeroshot_cot.GameOf24ZeroShotCoTPrompting):
    """The zeroshot prompt of GameOf24."""

    answer_content: str = "Answer: "
