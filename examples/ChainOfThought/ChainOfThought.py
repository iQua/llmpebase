"""
Implementation of Chain Of Thought [1].

[1]. Wei, et.al., Chain-of-Thought Prompting Elicits Reasoning in Large Language Models, 23.
"""

import reasoner

from llmpebase.pipeline import Pipeline

from llmpebase.config import Config


def _main():
    """The core function for model running."""
    model_config = Config().model
    model_config = Config.items_to_dict(Config().model._asdict())
    cot_reasoner = reasoner.CoTReasoner(model_config=model_config)

    pipeline = Pipeline(reasoner=cot_reasoner)
    pipeline.setup()
    pipeline.load_data()
    pipeline.execute()


if __name__ == "__main__":
    _main()
