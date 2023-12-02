"""
A formatter to convert the terminology to be the standard one,
i.e., A terminology contains one ore more words separated by 
white spaces, and the first letter of each word is capitalized.
"""


def format_term(terminology: str):
    """Format the terminology to be the standard one."""
    return terminology.replace("_", " ").replace("and", "&").rstrip().title()
