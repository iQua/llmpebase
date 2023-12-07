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
)


gt_extractors = {
    "GSM8K": {"re": GSM8KGtReExtractor, "llm": "not implemented"},
    "MMLU": {"re": MMLUGtReExtractor, "llm": "not implemented"},
}

resp_extractors = {
    "GSM8K": {"re": GSM8KRespReExtractor, "llm": "not implemented"},
    "MMLU": {"re": MMLURespReExtractor, "llm": "not implemented"},
}


def get(data_name, purpose: str = None, style: str = None, **kwargs):
    """Get the extractors for the specific dataset."""

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
