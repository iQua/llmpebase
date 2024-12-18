"""
The main running session of Boosting of Thoughts (BoT).
"""

import reasoner
import commenter
import thought_prompter
import thought_model
import bot_pipeline

from llmpebase.model import define_model
from llmpebase.prompt import (
    get_system_prompts,
    get_thought_prompts,
    get_chain_comment_prompts,
)
from llmpebase.pipeline import Pipeline
from llmpebase.config import Config
from llmpebase.model.thought_structure.visualization import BasicStructureVisualizer


def _main():
    """The core function for model running."""
    model_config = Config.items_to_dict(Config().model._asdict())
    logging_config = Config.items_to_dict(Config().logging._asdict())
    data_config = Config.items_to_dict(Config().data._asdict())

    # Define the necessary model
    llm_model = define_model(model_config=model_config)

    # Define the prompts for the thought structure
    system_prompts = get_system_prompts(data_config)
    thought_prompts = get_thought_prompts(data_config)
    chain_comment_prompts = get_chain_comment_prompts(
        comment_type=model_config["bot_settings"]["commenter"]["comment_type"],
        data_config=data_config,
    )

    # Define the thought model
    bot_thought_prompter = thought_prompter.BoTThoughtPrompter(
        system_prompts=system_prompts, thought_prompts=thought_prompts
    )
    bot_thought_model = thought_model.BoTThoughtModel(
        llm_model=llm_model, model_config=model_config, prompter=bot_thought_prompter
    )

    # Define the comment model
    # Note that as the llm model is shared between thought model and the
    # comment model, one may need to adjust the generation config
    # of the comment model.
    bot_comment_prompter = commenter.BoTCommentPrompter(
        system_prompts=system_prompts, comment_prompts=chain_comment_prompts
    )
    comment_model = commenter.BoTCommenter(
        llm_model=llm_model,
        model_config=model_config,
        logging_config=logging_config,
        prompter=bot_comment_prompter,
    )

    bot = reasoner.BoTReasoner(
        thought_model=bot_thought_model,
        comment_model=comment_model,
        model_config=model_config,
        logging_config=logging_config,
        visualizer=BasicStructureVisualizer(logging_config=logging_config),
    )

    pipeline = bot_pipeline.BoTPipeline(reasoner=bot)
    pipeline.setup()
    pipeline.load_data()
    pipeline.execute()


if __name__ == "__main__":
    _main()
