"""
A list of well-tested extracters used to extract sub-strings from the string.
"""

import re


def extract_final_sentence(string: str):
    """Extract the final sentence from the string."""
    pattern = r"[^.!\n\s]+(?:[.!\n]|$)"

    # Find all sentences in the paragraph
    sentences = re.findall(pattern, string)

    # Extract the final sentence
    return sentences[-1]


def extract_target_result(
    string: str,
    target_format="$",
):
    """Extract the target results from the string."""
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
    matches = re.findall(pattern, string)

    # Extract the matched numbers
    numbers = [match for match in matches if match]

    # Remove useless commas
    numbers = [number.replace(",", "") for number in numbers]
    return numbers
