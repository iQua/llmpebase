"""
An interface to postprocess the response from LLM. Its major responsibility is to 
extract the target result from the response.
"""

import logging

from llmpebase.extractor.base import BaseLlmExtractor
from llmpebase.extractor.re_extraction import *


from llmpebase.extractor.re_extraction import (
    GSM8KGtReExtractor,
    GSM8KRespReExtractor,
    MMLUGtReExtractor,
    MMLURespReExtractor,
    MATHGtReExtractor,
    MATHRespReExtractor,
    BBHGtReExtractor,
    BBHRespReExtractor,
    TheoremGtReExtractor,
    TheoremRespReExtractor,
    GameOf24RespReExtractor,
    AQUAGtReExtractor,
)

from llmpebase.extractor.llm_extraction import (
    MMLURespLlmExtractor,
    MATHGtLlmExtractor,
    GameOf24RespLlmExtractor,
    CSQARespLlmExtractor,
    AQUARespLlmExtractor,
)

# The extractors for extracting groundtruth from the data sample
# - not needed: no need to extract the groundtruth is either already
# provided or easy to obtain
# - not implemented: not implemented yet
gt_extractors = {
    "GSM8K": {"re": GSM8KGtReExtractor, "llm": "not needed"},
    "MMLU": {"re": MMLUGtReExtractor, "llm": "no needed"},
    "MATH": {"re": MATHGtReExtractor, "llm": MATHGtLlmExtractor},
    "BBH": {"re": BBHGtReExtractor, "llm": "not needed"},
    "TheoremQA": {"re": TheoremGtReExtractor, "llm": "not needed"},
    "AQUA": {"re": AQUAGtReExtractor, "llm": "not needed"},
}

resp_extractors = {
    "GSM8K": {"re": GSM8KRespReExtractor, "llm": BaseLlmExtractor},
    "MMLU": {"re": MMLURespReExtractor, "llm": MMLURespLlmExtractor},
    "MATH": {"re": MATHRespReExtractor, "llm": BaseLlmExtractor},
    "BBH": {"re": BBHRespReExtractor, "llm": BaseLlmExtractor},
    "TheoremQA": {"re": TheoremRespReExtractor, "llm": BaseLlmExtractor},
    "GameOf24": {"re": GameOf24RespReExtractor, "llm": GameOf24RespLlmExtractor},
    "CSQA": {"re": "not implemented", "llm": CSQARespLlmExtractor},
    "AQUA": {"re": "not implemented", "llm": AQUARespLlmExtractor},
    "SVAMP": {"re": "not implemented", "llm": BaseLlmExtractor},
}


def get(data_name, config: dict, **kwargs):
    """Get the extractors for the specific dataset."""
    purpose = config["purpose"]
    style = config["style"]

    assert purpose in ["groundtruth", "result"]
    assert style in ["re", "llm"]

    extractors = gt_extractors if purpose == "groundtruth" else resp_extractors

    if data_name in extractors:
        logging.info(
            "Get %s extractor for %s to extract %s",
            style,
            data_name,
            purpose,
        )
        return extractors[data_name][style]

    else:
        raise NotImplementedError(
            f"{style} extractor is not implemented for {data_name}"
        )
