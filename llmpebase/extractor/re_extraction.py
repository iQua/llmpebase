"""
A extractor relying on `re` of the python package to perform the extraction.
"""

import re

from llmpebase.extractor import base


def extract_sentences(text_str: str):
    """Extract the final sentence from the string."""

    # r"[^.!?\n]+[.!?](?=\s|$)"
    pattern = r"[.\n]+\s*"

    # Find all sentences in the paragraph
    sentences = re.split(pattern, text_str.rstrip())

    sentences = [sent for sent in sentences if not sent.isspace() and len(sent) != 0]

    # Extract the final sentence
    return sentences


def extract_target_answers(response_contents: list, solution_flag="In summary"):
    """Extracting the target answer from the contents of responses."""
    prefix = re.escape(f"{solution_flag}") + r".*"
    # 1. extract the string after the answer format
    pattern = rf"{prefix}\s*(.+)"

    obtained_targets = []
    for content in response_contents:
        match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)

        obtained_targets.append(match.group(0) if match else content)

    return obtained_targets


def extract_figures(
    text_str: str,
    target_format="$",
):
    """Extract the target results from the text_str."""
    # This pattent is used to extract the target result from the given
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

    # Extract the matched numbers
    numbers = [match for match in matches if match]

    # Remove useless commas
    numbers = [number.replace(",", "") for number in numbers]
    return numbers


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

    return matches


class GSM8KGtReExtractor(base.BaseReExtractor):
    """A base extractor to extract the groundtruth from the response."""

    def forward(self, answer):
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
        gt_sent = sentences[-1]

        # Extract the figures with `#` as the format, such as
        # ####7
        result = extract_figures(gt_sent, target_format="#")[-1]

        return answer, conclusion, result


class GSM8KRespReExtractor(base.BaseReExtractor):
    """A base extractor to extract the result from the response."""

    def forward(self, answer):
        """Extract the result from the response for the GSM8K dataset."""

        # Split the long string into separate sentences
        sentences = extract_sentences(answer)

        # Extract the corresponding answer, conclusion, and the sentence containing
        # the groundtruth
        conclusion = sentences[-2]
        gt_sent = sentences[-1]

        result = extract_figures(gt_sent, target_format="#")[-1]

        return answer, conclusion, result
