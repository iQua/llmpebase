"""
The main running session of optimizing the p-RAG.
"""

import pRAR_system_prompts
import pRAR_prompts
import thought_prompter
import thought_model
import reasoner
from optimize_pipeline import PlanOptimizationPipeline
from visualization import PRARVisualizer, node_config, edge_config

from llmpebase.model import define_model

# from llmpebase.prompt import get_system_prompts
from llmpebase.config import Config
from llmpebase.model.thought_structure.visualization import BasicStructureVisualizer


def _main():
    """The core function for model running."""
    model_config = Config.items_to_dict(Config().model._asdict())
    logging_config = Config.items_to_dict(Config().logging._asdict())
    data_config = Config.items_to_dict(Config().data._asdict())

    # Define the llm model
    llm_model = define_model(model_config=model_config)

    # Define the prompts for the thought structure
    system_prompts = pRAR_system_prompts.PlanSystemPrompts
    plan_prompts = pRAR_prompts.PlanPrompts
    plan_thought_prompts = pRAR_prompts.BasePlanThoughtPrompts

    # Define the thought model
    prar_thought_prompter = thought_prompter.PlanThoughtPrompter(
        system_prompts=system_prompts,
        thought_prompts=plan_thought_prompts,
        plan_prompts=plan_prompts,
    )

    prar_thought_model = thought_model.PlanThoughtModel(
        llm_model=llm_model, model_config=model_config, prompter=prar_thought_prompter
    )

    prar_reasoner = reasoner.PlanThoughtReasoner(
        thought_model=prar_thought_model,
        model_config=model_config,
        logging_config=logging_config,
        visualizer=PRARVisualizer(
            logging_config=logging_config,
            plot_config={"node_config": node_config, "edge_config": edge_config},
        ),
        solution_extractor=None,
    )

    pipeline = PlanOptimizationPipeline(reasoner=prar_reasoner)
    pipeline.setup()
    pipeline.load_data()
    pipeline.execute()


if __name__ == "__main__":
    _main()
