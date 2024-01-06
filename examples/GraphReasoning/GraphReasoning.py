"""
A reasoning process organized as a graph thought structure.
"""

import reasoner

from llmpebase.pipeline import Pipeline
from llmpebase.model import define_model
from llmpebase.model.thought_structure import thought_model

from llmpebase.config import Config


def _main():
    """The core function for model running."""
    # Set the basic llm model to be used by each component
    model_config = Config.items_to_dict(Config().model._asdict())
    logging_config = Config.items_to_dict(Config().logging._asdict())

    llm_model = define_model(model_config=model_config)

    llm_thought = thought_model.LlmThoughtModel(llm_model=llm_model)

    chain_reasoner = reasoner.GraphReasoner(
        thought_model=llm_thought,
        model_config=model_config,
        logging_config=logging_config,
    )

    pipeline = Pipeline(
        reasoner=chain_reasoner,
    )
    pipeline.setup()
    pipeline.load_data()
    pipeline.execute()


if __name__ == "__main__":
    _main()
