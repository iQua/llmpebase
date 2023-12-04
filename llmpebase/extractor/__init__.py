"""
An interface to postprocess the response from LLM. Its major responsibility is to 
extract the target result from the response.
"""


from llmpebase.extractor.re_extraction import GSM8KGtReExtractor, GSM8KRespReExtractor


gt_extractors = {"GSM8K": {"re": GSM8KGtReExtractor, "llm": "not implemented"}}

resp_extractors = {"GSM8K": GSM8KRespReExtractor}


def get(data_name, purpose: str = None, style: str = None, **kwargs):
    """Get the extractors for the specific dataset."""

    assert purpose in ["gt", "response"]
    assert style in ["re", "llm"]

    extractors = gt_extractors if purpose == "gt" else resp_extractors

    if data_name in extractors:
        return extractors[data_name][style]

    else:
        raise NotImplementedError(
            f"{style} extractor is not implemented for {data_name}"
        )
