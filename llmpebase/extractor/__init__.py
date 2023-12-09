"""
An interface to postprocess the response from LLM. Its major responsibility is to 
extract the target result from the response.
"""
import logging

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
)

from llmpebase.extractor.llm_extraction import GSM8KRespLlmExtractor

gt_extractors = {
    "GSM8K": {"re": GSM8KGtReExtractor, "llm": "not implemented"},
    "MMLU": {"re": MMLUGtReExtractor, "llm": "not implemented"},
    "MATH": {"re": MATHGtReExtractor, "llm": "not implemented"},
    "BBH": {"re": BBHGtReExtractor, "llm": "not implemented"},
    "TheoremQA": {"re": TheoremGtReExtractor, "llm": "not implemented"},
}

resp_extractors = {
    "GSM8K": {"re": GSM8KRespReExtractor, "llm": GSM8KRespLlmExtractor},
    "MMLU": {"re": MMLURespReExtractor, "llm": "not implemented"},
    "MATH": {"re": MATHRespReExtractor, "llm": "not implemented"},
    "BBH": {"re": BBHRespReExtractor, "llm": "not implemented"},
    "TheoremQA": {"re": TheoremRespReExtractor, "llm": "not implemented"},
    "GameOf24": {"re": GameOf24RespReExtractor, "llm": "not implemented"},
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
