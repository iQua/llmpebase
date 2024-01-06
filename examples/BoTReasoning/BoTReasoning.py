"""
The main running session of Boosting of Thoughts (BoT).
"""

import reasoner
import thought_model
import commenter
import thought_prompter

from llmpebase.model import define_model
from llmpebase.pipeline import Pipeline
from llmpebase.config import Config


def _main():
    """The core function for model running."""
    model_config = Config.items_to_dict(Config().model._asdict())
    logging_config = Config.items_to_dict(Config().logging._asdict())

    llm_model = define_model(model_config=model_config)
    bot_thought_prompter = thought_prompter.BoTThoughtPrompter()
    bot_thought_model = thought_model.BoTThoughtModel(
        llm_model=llm_model, model_config=model_config, prompter=bot_thought_prompter
    )

    bot_comment_prompter = commenter.BoTCommentPrompter()
    bot_commenter = commenter.BoTCommenter(
        llm_model=llm_model, model_config=model_config, prompter=bot_comment_prompter
    )

    bot = reasoner.BoTReasoner(
        thought_model=bot_thought_model,
        chain_commenter=bot_commenter,
        model_config=model_config,
        logging_config=logging_config,
    )

    pipeline = Pipeline(reasoner=bot)
    pipeline.setup()
    pipeline.load_data()
    pipeline.execute()


if __name__ == "__main__":
    _main()
