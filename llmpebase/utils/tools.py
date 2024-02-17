""" Tools used by the whole project."""


def format_term(terminology: str):
    """Format the terminology to be the standard one.
    This function ensure that all the terminology are in the same format.
    """
    # basic conversion
    terminology = (
        terminology.replace("_", " ").replace("and", "&").replace("-", " ").rstrip()
    )
    if terminology.isupper():
        return terminology

    return terminology.title()
