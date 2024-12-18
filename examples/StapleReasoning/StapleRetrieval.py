"""
The main running session of retrieving the p-RAG.
"""

import staple_system_prompts
import staple_prompts
import thought_prompter
import thought_model
import reasoner
from retrieval_pipeline import PlanRetrievalPipeline
from visualization import StapleVisualizer, node_config, edge_config

from llmpebase.model import define_model

# from llmpebase.prompt import get_system_prompts
from llmpebase.config import Config
from llmpebase.model.thought_structure.visualization import BasicStructureVisualizer


def _main():
    """The core function for model running."""
    model_config = Config.items_to_dict(Config().model._asdict())
    logging_config = Config.items_to_dict(Config().logging._asdict())
    # data_config = Config.items_to_dict(Config().data._asdict())

    # Define the llm model
    llm_model = define_model(model_config=model_config)

    # Define the prompts for the thought structure
    system_prompts = staple_system_prompts.PlanSystemPrompts
    plan_prompts = staple_prompts.PlanPrompts
    plan_thought_prompts = staple_prompts.BasePlanThoughtPrompts

    # Define the thought model
    staple_thought_prompter = thought_prompter.PlanThoughtPrompter(
        system_prompts=system_prompts,
        thought_prompts=plan_thought_prompts,
        plan_prompts=plan_prompts,
    )

    staple_thought_model = thought_model.PlanThoughtModel(
        llm_model=llm_model, model_config=model_config, prompter=staple_thought_prompter
    )

    staple_reasoner = reasoner.PlanThoughtReasoner(
        thought_model=staple_thought_model,
        model_config=model_config,
        logging_config=logging_config,
        visualizer=StapleVisualizer(
            logging_config=logging_config,
            plot_config={"node_config": node_config, "edge_config": edge_config},
        ),
        solution_extractor=None,
    )

    pipeline = PlanRetrievalPipeline(reasoner=staple_reasoner)
    pipeline.setup()
    pipeline.load_data()
    pipeline.execute()


if __name__ == "__main__":
    _main()
