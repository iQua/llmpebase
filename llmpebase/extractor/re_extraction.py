"""
A extractor relying on `regular expression (re)` of the python package to perform the extraction.
"""

import re
from typing import List

from llmpebase.extractor import base
import pandas as pd


def extract_sentences(text_str: str):
    """Extract the final sentence from the string."""
    # Set the split pattern of the regular expression
    pattern = r"(?<!\d)\.\s+|\n"

    # Find all sentences in the paragraph
    sentences = re.split(pattern, text_str.rstrip())

    sentences = [sent for sent in sentences if not sent.isspace() and len(sent) != 0]

    # Extract the final sentence
    return sentences


def extract_flagged_conclusion(
    text_str: str, flags: List[str] = None, weights: List[int] = None
):
    """
    Extract the conclusion containing the flags.

    This function will count the number of flags * weights in each sentence and return the
    one 1) containing the most important flags and 2) the final one when there are multiple
    """
    sentences = extract_sentences(text_str)
    # Count each sentence matches how many flags
    sentence_matched = []
    for sent in sentences:
        sentence_matched.append(
            sum([int(flag in sent) * weights[idx] for idx, flag in enumerate(flags)])
        )

    # Get the sentence with the most flags
    n_matched = len(sentence_matched)
    max_matched = max(sentence_matched)

    # Reverse the list in order to get the index of the final max values
    sentence_matched.reverse()
    # Here -1 as the index starts from 0
    index = n_matched - 1 - sentence_matched.index(max_matched)

    return sentences[index]


def extract_solution(text_str: str, solution_flag: str = "The solution is"):
    """
    Extract the solution presented after the 'solution_flag'.

    This function extracts the solution presented after the 'solution_flag' in the
    text_str. The solution_flag is case-insensitive.
    """
    # Set the finding pattern of the regular expression
    solution_flag = re.escape(solution_flag)
    pattern = rf"{solution_flag}\s*(.*)"

    # Search for the solution
    match = re.search(pattern, text_str, re.IGNORECASE | re.DOTALL)

    # Once nothing is extract, just return the original text_str
    if not match:
        return None

    return match.group(1)


def extract_figures(
    text_str: str,
    target_format="$",
):
    """Extract the target results from the text_str."""
    # This pattern is used to extract the target result from the given
    # `target_format`
    # For example, when target_format is $
    # This pattern can extract
    # $6$, $14$ -> 6, 14
    # $6.5$, $14.88$ -> 6.5, 14.88
    # $6, 7$ -> 6, 7
    # $6.5, 6.7$ -> 6.5, 6.7
    # $7.000.222$, $1000,00,0$ -> 7.000.222, 1000,00,0

    pattern = rf"\{target_format}?(\d[\d,.]*(?:\.\d*)?)\{target_format}?,?"

    # Find all matches in the text
    matches = re.findall(pattern, text_str)

    if not matches:
        return None

    # Extract the matched numbers
    numbers = [match for match in matches if match]

    # Remove useless commas
    numbers = [number.replace(",", "") for number in numbers]
    return numbers


def extract_characters(text_str: str):
    """Extract the solution presented after the 'solution_flag'."""
    pattern = r"\b[A-Za-z]\b"

    # Find all matches in the input string
    matches = re.findall(pattern, text_str)

    if not matches:
        return None

    characters = [match for match in matches if match]

    return characters


def extract_equations(text_str: str, target_format="$"):
    """Extract the target equation from the text_str."""
    target_format = "$"
    # Define a regular expression pattern to match the desired substrings
    pattern = rf"\{target_format}+(.*?)\{target_format}+"

    # Use re.findall() to find all matches
    matches = re.findall(pattern, text_str)
    # Extract the matched numbers
    numbers = [match for match in matches if match]

    # Once nothing is extract, just return the original text_str
    if not numbers:
        numbers = [text_str]

    return numbers


def extract_format_equations(
    text_str: str, equation_format="=", target_format="\\boxed"
):
    """Extract the equations within a format, such as the latex format."""
    # First extract the equation
    splitted_eq = text_str.split(equation_format)
    right_eq = splitted_eq[-1]

    # Extract the target result within the target_format
    # \\boxed, which is the format used in the MATH dataset
    pattern = rf"{re.escape(target_format)}{{((?:[^{{}}]+|{{[^{{}}]+}})+)}}"
    matches = re.findall(pattern, right_eq)

    if not matches:
        return None

    return matches


class GSM8KGtReExtractor(base.BaseReExtractor):
    """A base extractor to extract the groundtruth from the response."""

    def forward(self, answer, **kwargs):
        """Extract the groundtruth from samples of the GSM8K dataset.

        The answer in GSM8K sample will be:
            sent1\n sent2\n ####groundtruth
        """
        # Extract the sentences separately from the answer
        sentences = extract_sentences(answer)
        # Extract the corresponding answer, conclusion, and the sentence containing
        # the groundtruth
        answer = "\n".join(sentences[:-1])
        conclusion = sentences[-2]
        gt_sentence = sentences[-1]

        # Extract the figures with `#` as the format, such as
        # ####7
        groundtruth = extract_figures(gt_sentence, target_format="#")[-1]
        groundtruth = groundtruth if groundtruth is not None else gt_sentence
        return answer, conclusion, groundtruth


class GSM8KRespReExtractor(base.BaseReExtractor):
    """A base extractor to extract the result from the response."""

    def forward(self, answer, **kwargs):
        """Extract the result from the response for the GSM8K dataset."""
        # To obtain the target solution
        conclusion = extract_solution(answer, solution_flag=kwargs["solution_flag"])
        # When no target solution is obtained, we assume that the final sentence
        # will be the solution following the common behaviors.
        if conclusion is None:
            sentences = extract_sentences(answer)
            # Extract the corresponding conclusion which is the final sentence
            conclusion = sentences[-1]

        result = extract_figures(conclusion, target_format="$")[-1]
        result = result.rstrip(".") if result is not None else conclusion
        return result


class MMLUGtReExtractor(base.BaseReExtractor):
    """A base extractor to extract the groundtruth from the response."""

    def forward(self, answer: pd.DataFrame, **kwargs):
        """Extract the groundtruth from samples of the GSM8K dataset."""
        row_idx = kwargs["row_idx"]

        # Get the groundtruth in the corresponding row
        answer = answer.iloc[row_idx, -1]
        answer = f"{answer}"
        return answer, answer, answer


class MMLURespReExtractor(base.BaseReExtractor):
    """A base extractor to extract the result from the response."""

    def forward(self, answer, **kwargs) -> List[str]:
        """Extract the result from the response for the GSM8K dataset."""

        # To obtain the target solution
        conclusion = extract_solution(answer, solution_flag=kwargs["solution_flag"])

        # When no target solution is obtained, we assume that the final sentence
        # will be the solution following the common behaviors.
        if conclusion is None:
            sentences = extract_sentences(answer)
            # Extract the corresponding conclusion which is the final sentence
            conclusion = sentences[-1]

        results = extract_characters(conclusion)
        # Only maintain the A/B/C/D option as the MMLU is a single-choice dataset
        result = results[-1].rstrip(".") if results is not None else conclusion
        return result


class MATHGtReExtractor(base.BaseReExtractor):
    """A base extractor to extract the groundtruth from the response."""

    def forward(self, answer: str, **kwargs):
        """Extract the groundtruth from samples of the GSM8K dataset."""

        conclusion = extract_flagged_conclusion(
            answer, flags=["=", "\\boxed"], weights=[1, 3]
        )

        groundtruths = extract_format_equations(conclusion, target_format="\\boxed")

        groundtruth = groundtruths[-1]

        return answer, conclusion, groundtruth


class MATHRespReExtractor(base.BaseReExtractor):
    """A base extractor to extract the result from the response."""

    def forward(self, answer, **kwargs) -> List[str]:
        """Extract the result from the response for the GSM8K dataset."""

        # To obtain the target solution
        conclusion = extract_solution(answer, solution_flag=kwargs["solution_flag"])

        # When no target solution is obtained, we assume that the final sentence
        # will be the solution following the common behaviors.
        if conclusion is None:
            sentences = extract_sentences(answer)
            # Extract the corresponding conclusion which is the final sentence
            conclusion = sentences[-1]

        results = extract_equations(conclusion, target_format="$")
        # Only maintain the A/B/C/D option as the MMLU is a single-choice dataset
        result = results[-1].rstrip(".") if results is not None else conclusion
        return result
