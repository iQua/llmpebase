"""
A list of well-tested extracters used to extract sub-strings from the string.
"""

import re


def extract_sentences(text_str: str):
    """Extract the final sentence from the string."""

    # r"[^.!?\n]+[.!?](?=\s|$)"
    pattern = r"[.\n]+\s*"

    # Find all sentences in the paragraph
    sentences = re.split(pattern, text_str.rstrip())

    sentences = [sent for sent in sentences if not sent.isspace() and len(sent) != 0]

    # Extract the final sentence
    return sentences


def extract_target_result(
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


def extract_target_equations(text_str: str, target_format="$"):
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


def extract_equation_result(
    text_str: str, equation_format="=", target_format="\\boxed"
):
    """Extract the result from the equation."""
    # First extract the equation
    splitted_eq = text_str.split(equation_format)
    right_eq = splitted_eq[-1]

    # Extract the target result within the target_format
    # \\boxed, which is the format used in the MATH dataset
    pattern = rf"{re.escape(target_format)}{{((?:[^{{}}]+|{{[^{{}}]+}})+)}}"
    matches = re.findall(pattern, right_eq)

    return matches


def extract_compression_style(url):
    """Extract the style of compression from the url."""
    pattern = r"\.(zip|tar|tar\.gz)$"
    match = re.search(pattern, url)
    if match is not None:
        return match.group(1)

    return "zip"
