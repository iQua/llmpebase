"""
Thought structure for the NIPS submission.
"""

import logging
import math
import math
from typing import List

from plan_tree import PlanNode, PlanTree


from llmpebase.model.thought_structure import base
from llmpebase.model.thought_structure.structure_generic import BasicNode


def compute_explore_prob(num_nodes: int, constant: 5):
    """Compute the exploration probability."""
    # Compute the probability with the sigmoid function
    return 1 / (1 + math.exp(num_nodes - constant))


class NIPSPlanStructure(base.BaseThoughtStructure):
    """A reasoning method based on Monte Carlo tree search (MCTS)."""

    def __init__(
        self,
        thought_model,
        model_config,
        logging_config,
        visualizer,
    ):
        super().__init__(
            thought_model=thought_model,
            model_config=model_config,
            logging_config=logging_config,
            visualizer=visualizer,
        )

        # The plan tree to be used in the MCTS reasoning
        # Note that this tree should be adjusted based on the task to solve
        # meaning that each task should load its corresponding tree.
        self.plan_tree: PlanTree = None

        # Set the chains of the reasoning and plan
        self.reasoning_chain: List[BasicNode] = []
        self.plan_chain: List[PlanNode] = []

        # The settings for the MCTS reasoning.
        self.optimization_config = model_config["optimization"]
        self.mcts_config = self.optimization_config["mcts"]

        # Get the constant for the sigmoid function
        # to compute the probability of exploration
        self.explore_constant = self.mcts_config["plan_exploration_constant"]

        # The number of thoughts to be considered during the value computation
        self.num_neighbor_thoughts = self.mcts_config["num_neighbor_thoughts"]

    def expansion(
        self,
        cur_plan_node: PlanNode,
        cur_thought_node: BasicNode,
        thought_plan: str,
        next_thought_node: BasicNode,
    ):
        """Perform the expansion of MCTS."""

        # Add the new plan to the plan tree
        cur_thought_str = str(cur_thought_node.thought)
        if cur_thought_node.step_idx == 0:
            cur_thought_str = str(cur_thought_node.thought.question)
        plan_node_id = self.plan_tree.add_node(
            plan=thought_plan,
            plan_num_visits=1,
            prev_node_id=cur_plan_node.identity,
            thought_origins=[cur_thought_str],
            # Add the [num_wins, v_llm, n_visits]
            thought_evaluations=[[0, 0, 1]],
        )

        logging.info(
            "  Expanded new plan node N-%s as a child of node N-%s .",
            plan_node_id,
            cur_plan_node.identity,
        )

        return plan_node_id

    def reason_simulation(self):
        """Perform the simulation, i.e., reasoning, of the MCS."""
        # Continue reasoning until the sink node is reached
        # i.e., the solution is obtained
        # Create an empty plan tree

        logging.info(
            "Performing the simulation/rollout of MCTS from N-%s S-%s:",
            self.reasoning_chain[-1].identity,
            self.reasoning_chain[-1].step_idx,
        )

        while not self.is_node_sink(self.reasoning_chain[-1].identity):

            cur_thought_node = self.reasoning_chain[-1]
            cur_plan_node = self.plan_chain[-1]

            next_thoughts, infer_info = self.thought_model.generate_thoughts(
                thought_chain=self.reasoning_chain,
                num_thoughts=1,
                plan_chain=self.plan_chain,
            )

            next_thought = next_thoughts[0]
            infer_info = infer_info[0]
            logging.info(
                "  -> Generated next Step %s.",
                cur_thought_node.step_idx + 1,
            )

            # Add the thought to the thought structure as a node
            thought_node_id = self.add_node(
                thought=next_thought,
                prev_node_id=cur_thought_node.identity,
                thought_evaluation=None,
                thought_inference=infer_info,
            )
            # Add the thought node to the reasoning chain
            thought_node = self.node_pool[thought_node_id]
            # Summarize the plan of the thought when there is no plan
            # used during this thought generation

            # Summarize the plan from the generated thought
            new_plan, infer_info = self.thought_model.summarize_plan(
                thought_chain=self.reasoning_chain,
                plan_chain=self.plan_chain,
                thought_plan_node=self.node_pool[thought_node_id],
                num_thoughts=1,
            )
            new_plan = new_plan[0]

            plan_node_id = self.expansion(
                cur_plan_node=cur_plan_node,
                cur_thought_node=cur_thought_node,
                thought_plan=new_plan,
                next_thought_node=self.node_pool[thought_node_id],
            )
            selected_plan_node = self.plan_tree.node_pool[plan_node_id]
            self.plan_chain.append(selected_plan_node)

            self.reasoning_chain.append(thought_node)

            self.save_structure()
            self.plan_tree.save_structure(
                foldername="plan_tree", location=self.create_save_folder()
            )
            if self.visualizer is not None:
                self.visualizer.visualize(
                    self.graph,
                    self.node_pool,
                    save_name=f"Step_{thought_node.step_idx}",
                )

    def build_structure(
        self,
        **kwargs,
    ):
        """
        Perform the MCTS reasoning to update the plan tree.
        This corresponds to the algorithm table of the p-RAQ paper.

        Before running this function, it is desired to run the
        start_mcts and set_plan_tree functions.
        """

        # Make sets empty
        self.plan_chain: List[PlanNode] = [self.plan_tree.root]
        self.reasoning_chain: List[BasicNode] = [self.root]

        # Perform the simulation of the reasoning
        self.reason_simulation()

        # Draw the whole graph after building
        if self.visualizer is not None:
            self.visualizer.visualize(
                self.graph, self.node_pool, save_name="built_structure"
            )
            self.visualizer.visualize(
                self.plan_tree.graph,
                self.plan_tree.node_pool,
                save_name="plan_built_structure",
            )

    def reset_structure(self):
        """Reset the tee."""
        super().reset_structure()
        self.reasoning_chain: List[BasicNode] = []
        self.plan_chain: List[PlanNode] = []
