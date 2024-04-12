"""
Implementation of creating a format prompt with suitable indents.

For example, given a string namely EXP1:

<Experiences>
<Experience-1>
<Chains>
Step 1...
Step 2...
Step 3...
<\\Chains>
<\\Experience-1>
<\\Experiences>

Formatted string with suitable indents:
<Experiences>
    <Chains>
        Step 1...
        Step 2...
        Step 3...
    <\\Chains>
<\\Experiences>


Generally in Python, 
    - the backslash '\' has two purposes:
        1. To 'escape' literal characters (main and default).
          where some literal characters may have special meaning but they just look like the literal characters. The escape here means to escape the literal meaning but focus on the special meaning. For example, 'n' is a literal character but '\n' is a special character that represents a new line, where '\' is to escape n from the literal.

        2. To represent the real backslash string.

    - r'' is used to remove the special 'escape' property of the backslash '\'
      where r hear means the raw string.

    Thus, to to present backslash '\', one should use '\\' or r'\'.

In re, the backslash '\' is more complicated because it has two levels of responsibilities:
    - 'escape' literal characters from python interpreter level
    - 'escape' special characters from re level

    thus, to represent backslash '\', one should use '\\\\'. 
    The first level is the escape from python interpreter, leading to '\\' and the second level is the escape from re, leading to '\'.
    
    * But, it is recommended to use r'' to skip the python interpreter level escape.
    '\' can be represented as r'\\' in re.
"""

from typing import List, Tuple
import re
import textwrap


def create_tag_pattern(start_tag="<>", end_tag="<\\>"):
    """
    Generate the re pattern from the given tag, to extract the matched tag and the content inside.
    For example, the EXP1 given above, this pattern with re.findall will output
    [
        ('Experiences', ''\n<Experience-1>\n<Chain>...\n<\\Chain>\n<\\Experience-1>\n')
    ]
    """

    # Creating the start and end pattern
    open_tag_pattern = (
        re.escape(start_tag[0])
        + "([^"
        + re.escape(start_tag[-1])
        + "]+)"
        + re.escape(start_tag[-1])
    )
    # Note that re.escape here is same as adding 'r' before the string
    close_tag_pattern = re.escape(end_tag[0]) + "\\\\\\1" + re.escape(end_tag[-1])

    # Create the full regex pattern
    pattern = f"{open_tag_pattern}(.*?){close_tag_pattern}"

    return re.compile(pattern, re.DOTALL)


def replace_indent(input_str: str, cur_indent: str, target_indent: str):
    """Replace the cur_indent in input_str with the target_indent."""
    # Creating a regular expression pattern
    pattern = re.compile(f"^{cur_indent}|((?<=^{cur_indent}+){cur_indent})", flags=re.M)
    # Search and Replace
    input_str = re.sub(pattern, target_indent, input_str)

    return input_str


def format_prompt(
    prompt_str: str,
    start_tag: str = "<>",
    end_tag: str = "<\\>",
    indent: str = "  ",
    level: int = 1,
):
    """
    Format the contents in nested tags.

    :param prompt_str: The string to be formatted.
    :param start_tag: The start tag.
    :param end_tag: The end tag.
    :param indent: The indent to be added.
    :param level: How many indents to be added for contents of each matched tag.
    """

    # Create a pattern to match the paired tags
    pattern = create_tag_pattern(start_tag, end_tag)

    # Corrected regex pattern to handle the custom tags
    # tag_matches will contain the tag name and the content inside the tag
    tag_matches: List[Tuple[str, str]] = re.findall(pattern, prompt_str)

    for _, content in tag_matches:

        # Add the indent to the content
        indented_content = textwrap.indent(content, indent * level)

        # Recursively format the nested tags
        indented_content = format_prompt(
            indented_content, start_tag, end_tag, indent, level
        )
        # Replace the content with the indented content
        prompt_str = prompt_str.replace(content, indented_content)

    return prompt_str
