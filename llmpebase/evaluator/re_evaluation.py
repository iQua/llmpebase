"""
A base evaluator using `re` of Python of achieve the evaluation.
"""

from llmpebase.evaluator import base


def convert_item(item_str: str):
    """Convert the item to the target type."""
    item_str = item_str.strip()

    try:
        item_str = float(item_str)
    except ValueError:
        # if this is pure string, convert it
        # to the format case.
        item_str = item_str.lower()

    return item_str


def convert_str2float(input_str: str) -> float:
    """Convert the string to float."""
    if isinstance(input_str, str):
        # Remove the '.' in the final
        input_str = input_str.rstrip(".")

        # Process the string with [] in it
        if "[" in input_str and "]" in input_str and "," in input_str:
            input_str = input_str.replace("[", "").replace("]", "")
            input_str = input_str.split(",")
            input_str = [convert_item(elem) for elem in input_str]

        else:
            input_str = convert_item(input_str)

    elif isinstance(input_str, (list, tuple)):
        input_str = [convert_str2float(elem) for elem in input_str]

    return input_str


def do_conversion(input_str):
    """Perform the conversion."""
    input_str = convert_str2float(input_str)
    return input_str


class GeneralEvaluator(base.BaseEvaluator):
    """A base evaluator to perform tne measurement in a general way meaning that
    it will cover as many conditions as possible."""

    def measure(self, result: str, groundtruth: str):
        try:
            result = do_conversion(result) == do_conversion(groundtruth)
        except ValueError:
            result = None
        return result
