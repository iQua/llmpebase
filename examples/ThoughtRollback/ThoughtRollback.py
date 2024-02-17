"""
The implementation of the Thought Rollback (TR).
"""

import reasoner
import thought_model
import thought_prompter
import visualization

from llmpebase.pipeline import Pipeline
from llmpebase.model import define_model


from llmpebase.config import Config


def _main():
    """The core function for model running."""
    # Set the basic llm model to be used by each component
    model_config = Config.items_to_dict(Config().model._asdict())
    logging_config = Config.items_to_dict(Config().logging._asdict())

    llm_model = define_model(model_config=model_config)
    prompter = thought_prompter.TRStructurePrompt()
    llm_thought = thought_model.TRThoughtModel(
        llm_model=llm_model, model_config=model_config, prompter=prompter
    )

    tr_reasoner = reasoner.ThoughtRollbackReasoner(
        thought_model=llm_thought,
        model_config=model_config,
        logging_config=logging_config,
        visualizer=visualization.TRVisualizer(logging_config=logging_config),
    )

    pipeline = Pipeline(
        reasoner=tr_reasoner,
    )
    pipeline.setup()
    pipeline.load_data()
    pipeline.execute()


if __name__ == "__main__":
    _main()
