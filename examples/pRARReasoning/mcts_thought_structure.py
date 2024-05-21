"""
An implementation of MCTS for the prompting plan.

In this thought structure, each node specifically contains the 
prompting plan.
"""

import logging
import math
import json
import math
import random
from typing import List, Tuple

import networkx as nx
import numpy as np

from plan_tree import PlanNode, PlanTree
from embedder import GPTEmbedder

from llmpebase.model.thought_structure import base
from llmpebase.model.thought_structure.structure_generic import BasicNode


def compute_explore_prob(num_nodes: int, constant: 5):
    """Compute the exploration probability."""
    # Compute the probability with the sigmoid function
    return 1 / (1 + math.exp(num_nodes - constant))


class MCTSPlanStructure(base.BaseThoughtStructure):
    """A reasoning method based on Monte Carlo tree search (MCTS)."""

    def __init__(
        self,
        thought_model,
        model_config,
        logging_config,
        visualizer,
        embedder: GPTEmbedder,
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

        # The embedder to embed the thoughts
        self.embedder = embedder

    def set_plan_tree(self, plan_tree: PlanTree):
        """Set the plan tree to be used in the mcts."""
        self.plan_tree = plan_tree

    def save_inference(self, inference_info, filename):
        """Save the inference information."""
        # Get the save path of the thought structure
        save_path = self.create_save_folder()
        file_path = f"{save_path}/{filename}.json"

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(inference_info, file)

    def plan_exploration(
        self,
    ) -> Tuple[str, str]:
        """Explore the plan with the llm during reasoning."""

        # Plan exploration
        # Get the latest node in the plan chain
        cur_plan_node = self.plan_chain[-1]
        chile_plan_nodes = self.plan_tree.get_successor_nodes(cur_plan_node.identity)
        chile_plan_nodes = list(chile_plan_nodes)

        num_childs = len(chile_plan_nodes)
        logging.info(
            "Performing the plan exploration for plan node N-%s S-%s:",
            cur_plan_node.identity,
            cur_plan_node.step_idx,
        )
        if (
            num_childs > 0  # There should have plan candidates
            and compute_explore_prob(num_childs, self.explore_constant) > 0.5
        ):
            logging.info("  Generating next step by PlanExclusionReasoning.")
            # Generate the next thought by excluding the current policies
            # thereby allowing LLMs to explore new policies
            next_thoughts, infer_info = (
                self.thought_model.generate_excluded_plan_thoughts(
                    thought_chain=self.reasoning_chain,
                    num_thoughts=1,
                    plan_chain=self.plan_chain,
                    plan_exclusion_candidates=chile_plan_nodes,
                )
            )
            next_thought = next_thoughts[0]
            node_name = "PlanExclusionThought"
            node_position = "PlanExclusionIntermediate"
            step_name = "PlanExclusion"
            edge_type = "PlanExclusionReasoning"

        else:
            logging.info("  Generating next step by ThoughtGenerationReasoning.")
            # Directly generate the next thought
            next_thoughts, infer_info = self.thought_model.generate_thoughts(
                thought_chain=self.reasoning_chain,
                num_thoughts=1,
                plan_chain=self.plan_chain,
            )
            next_thought = next_thoughts[0]
            infer_info = infer_info[0]
            node_name = "NormalGenerationThought"
            node_position = "NormalGenerationIntermediate"
            step_name = "ThoughtGeneration"
            edge_type = "ThoughtGenerationReasoning"

        # Summarize the plan from the generated thought
        thought_id = self.add_node(
            thought=next_thought,
            prev_node_id=self.reasoning_chain[-1].identity,
            thought_evaluation=None,
            thought_inference=infer_info,
            node_name=node_name,
            step_name=step_name,
            position=node_position,
            growth="Growable",
            edge_type=edge_type,
        )
        new_plan, infer_info = self.thought_model.summarize_plan(
            thought_chain=self.reasoning_chain,
            plan_chain=self.plan_chain,
            plan_thought_node=self.node_pool[thought_id],
            num_thoughts=1,
        )
        new_plan = new_plan[0]

        plan_thought_id = self.add_node(
            thought=new_plan,
            prev_node_id=thought_id,
            step_idx=self.node_pool[thought_id].step_idx,
            thought_evaluation=None,
            thought_inference=infer_info,
            node_name="PlanSummarizationThought",
            step_name="PlanSummarization",
            position="PlanSummarizationIntermediate",
            growth="Growable",
            edge_type="PlanSummarizationReasoning",
        )
        logging.info("   Generated new plan by PlanSummarizationReasoning.")
        return plan_thought_id, thought_id

    def selection(self, cur_thought_node: BasicNode, cur_plan_node: PlanNode):
        """Perform the selection of MCTS."""
        # Visit the thought_origins of the plan node candidates
        logging.info("Performing the selection of MCTS.")

        plan_node_candidates = self.plan_tree.get_successor_nodes(
            cur_plan_node.identity
        )
        candidate_scores = [
            self.compute_values(thought_node=cur_thought_node, plan_node=plan_node)
            for plan_node in plan_node_candidates
        ]
        candidate_scores = [sum(score) for score in candidate_scores]

        best_index = np.argmax(candidate_scores)

        select_plan_node = plan_node_candidates[best_index]
        logging.info(
            "  Selected the best plan N-%s S-%s for step %s",
            select_plan_node.identity,
            select_plan_node.step_idx,
            cur_thought_node.step_idx,
        )

        return select_plan_node

    def expansion(
        self,
        cur_plan_node: PlanNode,
        cur_thought_node: BasicNode,
        plan_thought_node: BasicNode,
        next_thought_node: BasicNode,
    ):
        """Perform the expansion of MCTS."""
        # Assess the thought of the plan
        # Correspond to the I_A of the p-RAR paper
        thoughts, infer_info = self.thought_model.assess_thought_plan(
            thought_chain=self.reasoning_chain,
            plan_thought_node=plan_thought_node,
            thought_node=next_thought_node,
        )
        try:
            llm_value = float(thoughts[0])
        except ValueError:
            llm_value = 0.0

        # Add the assessment to the thought structure
        self.add_node(
            thought=llm_value,
            prev_node_id=plan_thought_node.identity,
            step_idx=plan_thought_node.step_idx,
            thought_evaluation=None,
            thought_inference=infer_info[0],
            node_name="PlanAssessmentThought",
            position="PlanAssessmentIntermediate",
            step_name="PlanAssessment",
            growth="Un-growable",
            edge_type="PlanAssessmentReasoning",
        )

        # Add the new plan to the plan tree
        cur_thought_str = str(cur_thought_node.thought)
        if cur_thought_node.step_idx == 0:
            cur_thought_str = str(cur_thought_node.thought.question)
        plan_node_id = self.plan_tree.add_node(
            plan=plan_thought_node.thought,
            plan_num_visits=1,
            prev_node_id=cur_plan_node.identity,
            thought_origins=[cur_thought_str],
            # Add the [num_wins, v_llm, n_visits]
            thought_evaluations=[[0, llm_value, 1]],
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
        logging.info(
            "Performing the simulation/rollout of MCTS from N-%s S-%s:",
            self.reasoning_chain[-1].identity,
            self.reasoning_chain[-1].step_idx,
        )

        while not self.is_node_sink(self.reasoning_chain[-1].identity):

            cur_thought_node = self.reasoning_chain[-1]
            cur_plan_node = self.plan_chain[-1]
            plan_node_candidates = self.plan_tree.get_successor_nodes(
                cur_plan_node.identity
            )

            if len(plan_node_candidates) == 0:

                next_thoughts, infer_info = self.thought_model.generate_thoughts(
                    thought_chain=self.reasoning_chain,
                    num_thoughts=1,
                    plan_chain=self.plan_chain,
                )
                next_thought = next_thoughts[0]
                infer_info = infer_info[0]
                logging.info(
                    "  -> Generated next Step %s without plan guidance due to no plan candidates.",
                    cur_thought_node.step_idx + 1,
                )
                # Note that the plan of this thought should be summarized
                # For convenience, we summarize the plan of the thought
                # below.
            else:

                # Randomly select a plan to guide the reasoning
                selected_plan_node = random.choice(plan_node_candidates)

                next_thoughts, infer_info = (
                    self.thought_model.generate_plan_next_thoughts(
                        thought_chain=self.reasoning_chain,
                        plan_chain=self.plan_chain,
                        plan_node=selected_plan_node,
                        num_thoughts=1,
                    )
                )
                next_thought = next_thoughts[0]
                infer_info = infer_info[0]
                # Add the plan to the plan chain
                self.plan_chain.append(selected_plan_node)
                logging.info(
                    "  -> Generating next Step %s with plan N-%s S-%s guidance.",
                    cur_thought_node.step_idx + 1,
                    selected_plan_node.identity,
                    selected_plan_node.step_idx,
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
            if len(plan_node_candidates) == 0:
                # Summarize the plan from the generated thought
                new_plan, infer_info = self.thought_model.summarize_plan(
                    thought_chain=self.reasoning_chain,
                    plan_chain=self.plan_chain,
                    plan_thought_node=self.node_pool[thought_node_id],
                    num_thoughts=1,
                )
                new_plan = new_plan[0]
                plan_thought_id = self.add_node(
                    thought=new_plan,
                    prev_node_id=thought_node_id,
                    step_idx=self.node_pool[thought_node_id].step_idx,
                    thought_evaluation=None,
                    thought_inference=infer_info,
                    node_name="PlanSummarizationThought",
                    step_name="PlanSummarization",
                    position="PlanSummarizationIntermediate",
                    growth="Growable",
                    edge_type="PlanSummarizationReasoning",
                )
                plan_node_id = self.expansion(
                    cur_plan_node=cur_plan_node,
                    cur_thought_node=cur_thought_node,
                    plan_thought_node=self.node_pool[plan_thought_id],
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

    def backpropagation(self, is_win: int):
        """Record the correctness of the solution to the plan chain."""
        logging.info("Performing the backpropagation of MCTS:")
        for plan_node in self.plan_chain:
            node_id = plan_node.identity
            if plan_node.position == "PlanRoot":
                continue
            self.plan_tree.node_pool[node_id].plan_num_visits += 1
            # The latest thought is the one newly added for the plan
            self.plan_tree.node_pool[node_id].thought_evaluations[-1][0] += is_win
            logging.info(
                "  -> Updated the plan node N-%s S-%s.",
                node_id,
                self.plan_tree.node_pool[node_id].step_idx,
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
        if self.plan_tree is None:
            raise ValueError("The plan tree is not set.")

        # Make sets empty
        self.plan_chain: List[PlanNode] = [self.plan_tree.root]
        self.reasoning_chain: List[BasicNode] = [self.root]

        # Jump into the mcts loop toward updating the plan tree
        while not self.is_node_sink(self.reasoning_chain[-1].identity):
            # Get the current plan and the current thought
            # The current thought derives from the current plan
            cur_plan_node = self.plan_chain[-1]
            cur_thought_node = self.reasoning_chain[-1]

            # Perform the plan exploration of the p-RAR paper
            # Note that this process will be two nodes to the thought structure
            # 1. Added a thought node, either exploration or normal generation
            # 2. Added a plan summarization thought
            plan_thought_id, next_thought_id = self.plan_exploration()

            # Check whether the plan exists in the child of
            # the current plan node
            plan_node_candidates = self.plan_tree.get_successor_nodes(
                cur_plan_node.identity
            )

            is_exist = "false"
            if len(plan_node_candidates) > 0:
                logging.info(
                    "Judging whether the new plan exists in successors of N-%s.",
                    cur_plan_node.identity,
                )
                exist_thoughts, infer_info = self.thought_model.compare_plan(
                    target_plan_thought_node=self.node_pool[plan_thought_id],
                    plan_node_pool=plan_node_candidates,
                )
                is_exist = exist_thoughts[0]
                infer_info = infer_info[0]
                # Add the plan existence comparison to the thought structure
                self.add_node(
                    thought=is_exist,
                    prev_node_id=plan_thought_id,
                    step_idx=self.node_pool[plan_thought_id].step_idx,
                    thought_evaluation=None,
                    thought_inference=infer_info,
                    node_name="PlanExistenceThought",
                    position="PlanExistenceIntermediate",
                    step_name="PlanExistenceComparison",
                    growth="Un-growable",
                    edge_type="PlanExistenceReasoning",
                )

            if "true" in is_exist.lower():
                logging.info("Selecting best plan as the new plan exists.")
                # Select the best current plan to guide the reasoning
                best_plan_node = self.selection(
                    cur_thought_node=cur_thought_node,
                    cur_plan_node=cur_plan_node,
                )
                # Generate the next thought based on the selected plan
                thoughts, infer_info = self.thought_model.generate_plan_next_thoughts(
                    thought_chain=self.reasoning_chain,
                    plan_chain=self.plan_chain,
                    plan_node=best_plan_node,
                    num_thoughts=1,
                )
                next_thought = thoughts[0]
                infer_info = infer_info[0]
                # Add the thought to the thought structure and also the reasoning chain
                next_thought_id = self.add_node(
                    thought=next_thought,
                    prev_node_id=cur_thought_node.identity,
                    thought_evaluation=None,
                    thought_inference=infer_info,
                )
                # Append the plan
                self.plan_chain.append(best_plan_node)

                # Add the current thought to the plan
                next_thought_node = self.node_pool[next_thought_id]
                value_thoughts, infer_info = self.thought_model.assess_thought_plan(
                    thought_chain=self.reasoning_chain,
                    thought_node=next_thought_node,
                    plan_node=best_plan_node,
                )
                llm_value = float(value_thoughts[0])
                infer_info = infer_info[0]
                self.add_node(
                    thought=llm_value,
                    prev_node_id=next_thought_node.identity,
                    step_idx=next_thought_node.step_idx,
                    thought_evaluation=None,
                    thought_inference=infer_info,
                    node_name="PlanAssessmentThought",
                    position="PlanAssessmentIntermediate",
                    step_name="PlanAssessment",
                    growth="Un-growable",
                    edge_type="PlanAssessmentReasoning",
                )
                # Extend the node by adding the thought origin
                # and the thought evaluation
                cur_thought_str = str(cur_thought_node.thought)
                if cur_thought_node.step_idx == 0:
                    cur_thought_str = str(cur_thought_node.thought.question)
                self.plan_tree.extend_node(
                    node_id=best_plan_node.identity,
                    thought_origin=cur_thought_str,
                    # Add the [num_wins, v_llm, n_visits]
                    thought_evaluation=[0, llm_value, 1],
                )
            else:
                logging.info("Performing expansion of MCTS:")
                added_plan_node_id = self.expansion(
                    cur_plan_node=cur_plan_node,
                    cur_thought_node=cur_thought_node,
                    plan_thought_node=self.node_pool[plan_thought_id],
                    next_thought_node=self.node_pool[next_thought_id],
                )
                # Add the new plan to the plan chain
                new_plan_node = self.plan_tree.node_pool[added_plan_node_id]
                self.plan_chain.append(new_plan_node)

            # Add the next thought to the reasoning chain
            thought_node = self.node_pool[next_thought_id]
            self.reasoning_chain.append(thought_node)

            # Save the structure to the disk
            # Save the updated plan tree to the same folder
            # of the thought structure
            self.save_structure()
            self.plan_tree.save_structure(
                foldername="plan_tree", location=self.create_save_folder()
            )

            # Save the plan tree to the same folder
            if self.visualizer is not None:
                self.visualizer.visualize(
                    self.graph,
                    self.node_pool,
                    save_name=f"Step_{thought_node.step_idx}",
                )
                self.visualizer.visualize(
                    self.plan_tree.graph,
                    self.plan_tree.node_pool,
                    save_name=f"Plan Step_{self.plan_chain[-1].step_idx}",
                )

            # Jump out of the loop if a new plan is added
            # to the plan tree
            # Thus, in MCTS, we move to the simulation
            if "false" in is_exist.lower():
                logging.info("After expansion, jumping out toward Simulation.")
                break

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

    def compute_values(self, thought_node: BasicNode, plan_node: PlanNode):
        """Compute the value of a plan."""
        # A list, each is string of the thought origin
        thought_origins = plan_node.thought_origins
        # A list, each is (num_wins, v_llm)
        thought_evaluations = plan_node.thought_evaluations

        # Get the indexes of the K-nearest thoughts
        query = (
            str(thought_node.thought)
            if thought_node.step_idx > 0
            else str(thought_node.thought.question)
        )
        top_idx, distances = self.embedder.get_neighbors(
            query=query,
            documents=thought_origins,
            num_neighbors=self.num_neighbor_thoughts,
        )

        neighbor_evals = [thought_evaluations[idx] for idx in top_idx]
        # Compute the averaged win rate and llm
        # As each thought origin has the format of (num_wins, v_llm, n_visits)
        # we should compute the inner average before computing the outer average
        avg_win = np.mean([score[0] / score[2] for score in neighbor_evals])
        v_llm = np.mean([score[1] / score[2] for score in neighbor_evals])

        # Compute the average distance
        v_u = 1 / np.mean(distances)
        v_u = 1 if math.isinf(v_u) else v_u
        return avg_win, v_llm, v_u

    def compute_ucb_value(
        self,
        node: BasicNode,
        current_player: int = 1,
        constant_efficiency: float = None,
    ):
        """
        Compute the UCB value for the node.
        V_ucb(n_i) = Q(n_i) + C * sqrt(log(N) / n(n_i))
        where n_i is the node id while the n(n_i) represents the number of visits to the node.
        """
        # Get the parent node of the node
        parent_node = self.graph.predecessors(node.identity)
        parent_n_visits = parent_node.auxiliary["n_visits"]
        n_visits = node.auxiliary["n_visits"]
        # Get the total rewards from these child nodes
        total_reward = node.auxiliary["reward"]
        mean_reward = total_reward / n_visits

        # Compute the UCB value in which the current_player is the player who
        # is currently making the decision.
        constant_efficiency = (
            1 / math.sqrt(2) if constant_efficiency is None else constant_efficiency
        )
        return current_player * mean_reward + constant_efficiency * math.sqrt(
            2 * math.log(parent_n_visits) / n_visits
        )

    def reset_structure(self):
        """Reset the tee."""
        super().reset_structure()
        self.plan_tree = None
        self.reasoning_chain: List[BasicNode] = []
        self.plan_chain: List[PlanNode] = []
