"""
The implementation of applying BoT on the Game of 24.
"""
import logging
import random
from typing import List
from collections import OrderedDict

import BoT_reasoner
import BoT_aggregator


from llmpebase.models.prompting import tree_thoughts


class BoostOfThoughts:
    """A base class for the Boosting of Thoughts.

    To save space, we define one request model, one experience reasoner,
    one chain commenter to be used for all trees. For different trees,
    only the specific generation config and experiences are switched.
    """

    def __init__(self, experience_reasoner, chain_commenter, model_config) -> None:
        self.model_config = model_config
        # The core generation config
        self.base_generation_config = self.model_config["generation_settings"]
        # The number of iteration to be performed
        self.n_iteration = self.model_config["n_iteration"]

        # Set the BoT's experience reasoner
        self.experience_reasoner = experience_reasoner
        # Define the BoT's commenter that produces experience
        self.chain_commenter = chain_commenter

        # BoT's aggregator
        self.aggregator = BoT_aggregator.ReasoningChainAggregator(
            model_config=model_config
        )

        # Tree Heterogeneity
        n_trees = self.model_config["num_trees"]
        self.heterogeneity_trees = OrderedDict()
        self.tree_generation_config = OrderedDict()
        self.global_experience = OrderedDict()

        for tree_id in range(n_trees):
            tree_types = {
                "levelwise": tree_thoughts.RTTLevelWise,
                "levelwisebest": tree_thoughts.RTTLevelWiseBest,
                "leafwise": tree_thoughts.RTTLeafWise,
            }
            generation_config = self.base_generation_config
            # Set the first tree to be the base generation in the
            # config file
            selected_type = "levelwise"
            if tree_id > 1:
                tree_temperature = random.choice([0.2, 0.4, 0.6, 0.7, 0.9, 1.1, 1.5])
                tree_top_p = random.choice([0.1, 0.3, 0.5, 0.7, 0.9])
                generation_config = self.model_config["generation_settings"]
                generation_config["temperature"] = tree_temperature
                generation_config["top_p"] = tree_top_p

                selected_type = random.choice(list(tree_types.keys()))

            self.tree_generation_config[tree_id] = generation_config

            self.heterogeneity_trees[tree_id] = tree_types[selected_type](
                self.experience_reasoner,
                n_child_nodes=self.model_config["tree_settings"]["n_child_nodes"],
                model_config=self.model_config,
            )
            logging.info(
                "Built %s-th %s tree with generation config %s ",
                tree_id + 1,
                selected_type,
                generation_config,
            )

    def perform_local_reasoning(
        self, task_prompt: str, tree_id: int
    ) -> List[tree_thoughts.ThoughtNode]:
        """Perform the reasoning locally on each tree."""
        local_tree = self.heterogeneity_trees[tree_id]
        generation_config = self.tree_generation_config[tree_id]
        # Clean the old tree
        local_tree.reset_tree()
        # Update the generation config for this tree
        local_tree.model.request_model.update_generation_config(generation_config)

        # Add the initial task prompt to the root node
        local_tree.construct_tree_root(
            thought=task_prompt,
            thought_score=None,
        )
        # Build the thought tree to perform the reasoning
        local_tree.build_thought_tree()

        # Get the best chain which is a list of thought nodes
        best_chain, _ = local_tree.get_best_thought_chain()

        return best_chain

    def perform_global_aggregation(
        self, task_prompt: str, local_chains: List[tree_thoughts.ThoughtNode]
    ):
        """Perform the global ggregation for the local chains."""
        aggregated_chain = self.aggregator.perform_aggregation(chains=local_chains)

        # Evaluate the aggregated chain to get the experience
        # Convert the chain to the prompt
        # Update the config to be the core oen
        self.experience_reasoner.request_model.update_generation_config(
            self.base_generation_config
        )
        # Get the chain prompt
        chain_prompt = (
            BoT_reasoner.ExperienceRecallReasoner.organize_though_chain_prompt(
                node_thought_chain=aggregated_chain
            )
        )

        # Get the feedback
        feedback_content = self.chain_commenter.get_thought_chain_feedback(
            task_prompt, reasoning_chain_content=chain_prompt
        )
        comment_feedback, chain_prompt = feedback_content

        experience = BoT_reasoner.ExperienceRecallReasoner.create_experience(
            comment_feedback, chain_prompt
        )

        self.experience_reasoner.memory_experience(experience)

        return aggregated_chain

    def perform_bot_reasoning(
        self,
        task_prompt,
    ):
        """Perform the reasoning by using BoT with multiple trees and multiple iterations."""

        aggregated_chain = None
        for _ in range(self.n_iteration):
            # Perform the local reasoning on each tree
            local_chains = {}
            for tree_id in self.heterogeneity_trees:
                best_chain = self.perform_local_reasoning(task_prompt, tree_id)
                local_chains[tree_id] = best_chain
            # Perform the global aggregation
            aggregated_chain = self.perform_global_aggregation(
                task_prompt, local_chains
            )
            print(self.heterogeneity_trees[0].model.experiences)

        return aggregated_chain
