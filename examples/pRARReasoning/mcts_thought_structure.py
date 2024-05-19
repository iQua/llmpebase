"""
An implementation of MCTS for the prompting policy.

In this thought structure, each node specifically contains the 
prompting policy.
"""

import logging
import math
import json
import math
import random
from typing import List, Tuple

import networkx as nx
import numpy as np

from policy_tree import PolicyNode, PolicyTree
from embedder import GPTEmbedder

from llmpebase.model.thought_structure import base
from llmpebase.model.thought_structure.structure_generic import BasicNode


def compute_explore_prob(num_nodes: int, constant: 5):
    """Compute the exploration probability."""
    # Compute the probability with the sigmoid function
    return 1 / (1 + math.exp(num_nodes - constant))


class MCTSPolicyStructure(base.BaseThoughtStructure):
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

        # The policy tree to be used in the MCTS reasoning
        # Note that this tree should be adjusted based on the task to solve
        # meaning that each task should load its corresponding tree.
        self.policy_tree: PolicyTree = None

        # Set the chains of the reasoning and policy
        self.reasoning_chain: List[BasicNode] = []
        self.policy_chain: List[PolicyNode] = []

        # The settings for the MCTS reasoning.
        self.optimization_config = model_config["optimization"]
        self.mcts_config = self.optimization_config["mcts"]

        # Get the constant for the sigmoid function
        # to compute the probability of exploration
        self.explore_constant = self.mcts_config["policy_exploration_constant"]

        # The number of thoughts to be considered during the value computation
        self.num_neighbor_thoughts = self.mcts_config["num_neighbor_thoughts"]

        # The embedder to embed the thoughts
        self.embedder = embedder

    def set_policy_tree(self, policy_tree: PolicyTree):
        """Set the policy tree to be used in the mcts."""
        self.policy_tree = policy_tree

    def save_inference(self, inference_info, filename):
        """Save the inference information."""
        # Get the save path of the thought structure
        save_path = self.create_save_folder()
        file_path = f"{save_path}/{filename}.json"

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(inference_info, file)

    def policy_exploration(
        self,
    ) -> Tuple[str, str]:
        """Explore the policy with the llm during reasoning."""

        # Policy exploration
        # Get the latest node in the policy chain
        cur_policy_node = self.policy_chain[-1]
        chile_policy_nodes = self.policy_tree.get_successor_nodes(
            cur_policy_node.identity
        )
        chile_policy_nodes = list(chile_policy_nodes)

        num_childs = len(chile_policy_nodes)
        logging.info(
            "Performing the policy exploration for policy node N-%s S-%s:",
            cur_policy_node.identity,
            cur_policy_node.step_idx,
        )
        if compute_explore_prob(num_childs, self.explore_constant) > 0.5:
            logging.info("  Generating next step by PolicyExclusionReasoning.")
            # Generate the next thought by excluding the current policies
            # thereby allowing LLMs to explore new policies
            next_thoughts, infer_info = (
                self.thought_model.generate_excluded_policy_thoughts(
                    thought_chain=self.reasoning_chain,
                    num_thoughts=1,
                    policy_chain=self.policy_chain,
                    policy_exclusion_candidates=chile_policy_nodes,
                )
            )
            next_thought = next_thoughts[0]
            node_name = "PolicyExclusionThought"
            node_position = "PolicyExclusionIntermediate"
            step_name = "PolicyExclusion"
            edge_type = "PolicyExclusionReasoning"

        else:
            logging.info("  Generating next step by ThoughtGenerationReasoning.")
            # Directly generate the next thought
            next_thoughts, infer_info = self.thought_model.generate_thoughts(
                thought_chain=self.reasoning_chain,
                num_thoughts=1,
                policy_chain=self.policy_chain,
            )
            next_thought = next_thoughts[0]
            infer_info = infer_info[0]
            node_name = "NormalGenerationThought"
            node_position = "NormalGenerationIntermediate"
            step_name = "ThoughtGeneration"
            edge_type = "ThoughtGenerationReasoning"

        # Summarize the policy from the generated thought
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
        new_policy, infer_info = self.thought_model.summarize_policy(
            thought_chain=self.reasoning_chain,
            policy_chain=self.policy_chain,
            policy_thought_node=self.node_pool[thought_id],
            num_thoughts=1,
        )
        new_policy = new_policy[0]

        policy_thought_id = self.add_node(
            thought=new_policy,
            prev_node_id=thought_id,
            step_idx=self.node_pool[thought_id].step_idx,
            thought_evaluation=None,
            thought_inference=infer_info,
            node_name="PolicySummarizationThought",
            step_name="PolicySummarization",
            position="PolicySummarizationIntermediate",
            growth="Growable",
            edge_type="PolicySummarizationReasoning",
        )
        logging.info("   Generated new policy by PolicySummarizationReasoning.")
        return policy_thought_id, thought_id

    def selection(self, cur_thought_node: BasicNode, cur_policy_node: PolicyNode):
        """Perform the selection of MCTS."""
        # Visit the thought_origins of the policy node candidates
        logging.info("Performing the selection of MCTS.")

        policy_node_candidates = self.policy_tree.get_successor_nodes(
            cur_policy_node.identity
        )
        candidate_scores = [
            self.compute_values(thought_node=cur_thought_node, policy_node=policy_node)
            for policy_node in policy_node_candidates
        ]
        candidate_scores = [sum(score) for score in candidate_scores]

        best_index = np.argmax(candidate_scores)

        select_policy_node = policy_node_candidates[best_index]
        logging.info(
            "  Selected the best policy N-%s S-%s for step %s",
            select_policy_node.identity,
            select_policy_node.step_idx,
            cur_thought_node.step_idx,
        )

        return select_policy_node

    def expansion(
        self,
        cur_policy_node: PolicyNode,
        cur_thought_node: BasicNode,
        policy_thought_node: BasicNode,
        next_thought_node: BasicNode,
    ):
        """Perform the expansion of MCTS."""
        # Assess the thought of the policy
        # Correspond to the I_A of the p-RAR paper
        thoughts, infer_info = self.thought_model.assess_thought_policy(
            thought_chain=self.reasoning_chain,
            policy_thought_node=policy_thought_node,
            thought_node=next_thought_node,
        )
        try:
            llm_value = float(thoughts[0])
        except ValueError:
            llm_value = 0.0

        # Add the assessment to the thought structure
        self.add_node(
            thought=llm_value,
            prev_node_id=policy_thought_node.identity,
            step_idx=policy_thought_node.step_idx,
            thought_evaluation=None,
            thought_inference=infer_info[0],
            node_name="PolicyAssessmentThought",
            position="PolicyAssessmentIntermediate",
            step_name="PolicyAssessment",
            growth="Un-growable",
            edge_type="PolicyAssessmentReasoning",
        )

        # Add the new policy to the policy tree
        cur_thought_str = str(cur_thought_node.thought)
        if cur_thought_node.step_idx == 0:
            cur_thought_str = str(cur_thought_node.thought.question)
        policy_node_id = self.policy_tree.add_node(
            policy=policy_thought_node.thought,
            policy_num_visits=1,
            prev_node_id=cur_policy_node.identity,
            thought_origins=[cur_thought_str],
            # Add the [num_wins, v_llm, n_visits]
            thought_evaluations=[[0, llm_value, 1]],
        )

        logging.info(
            "  Expanded new policy node N-%s as a child of node N-%s .",
            policy_node_id,
            cur_policy_node.identity,
        )

        return policy_node_id

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
            cur_policy_node = self.policy_chain[-1]
            policy_node_candidates = self.policy_tree.get_successor_nodes(
                cur_policy_node.identity
            )

            if len(policy_node_candidates) == 0:

                next_thoughts, infer_info = self.thought_model.generate_thoughts(
                    thought_chain=self.reasoning_chain,
                    num_thoughts=1,
                    policy_chain=self.policy_chain,
                )
                next_thought = next_thoughts[0]
                infer_info = infer_info[0]
                logging.info(
                    "  -> Generated next Step %s without policy guidance due to no policy candidates.",
                    cur_thought_node.step_idx + 1,
                )
            else:

                # Randomly select a policy to guide the reasoning
                selected_policy_node = random.choice(policy_node_candidates)

                next_thoughts, infer_info = (
                    self.thought_model.generate_policy_next_thoughts(
                        thought_chain=self.reasoning_chain,
                        policy_chain=self.policy_chain,
                        policy_node=selected_policy_node,
                        num_thoughts=1,
                    )
                )
                next_thought = next_thoughts[0]
                infer_info = infer_info[0]
                # Add the policy to the policy chain
                self.policy_chain.append(selected_policy_node)
                logging.info(
                    "  -> Generating next Step %s with policy N-%s S-%s guidance.",
                    cur_thought_node.step_idx + 1,
                    selected_policy_node.identity,
                    selected_policy_node.step_idx,
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
            self.reasoning_chain.append(thought_node)

            self.save_structure()
            self.policy_tree.save_structure(
                foldername="policy_tree", location=self.create_save_folder()
            )
            if self.visualizer is not None:
                self.visualizer.visualize(
                    self.graph,
                    self.node_pool,
                    save_name=f"Step_{thought_node.step_idx}",
                )

    def backpropagation(self, is_win: int):
        """Record the correctness of the solution to the policy chain."""
        logging.info("Performing the backpropagation of MCTS:")
        for policy_node in self.policy_chain:
            node_id = policy_node.identity
            if policy_node.position == "PolicyRoot":
                continue
            self.policy_tree.node_pool[node_id].policy_num_visits += 1
            # The latest thought is the one newly added for the policy
            self.policy_tree.node_pool[node_id].thought_evaluations[-1][0] += is_win
            logging.info(
                "  -> Updated the policy node N-%s S-%s.",
                node_id,
                self.policy_tree.node_pool[node_id].step_idx,
            )

    def build_structure(
        self,
        **kwargs,
    ):
        """
        Perform the MCTS reasoning to update the policy tree.
        This corresponds to the algorithm table of the p-RAQ paper.

        Before running this function, it is desired to run the
        start_mcts and set_policy_tree functions.
        """
        if self.policy_tree is None:
            raise ValueError("The policy tree is not set.")

        # Make sets empty
        self.policy_chain: List[PolicyNode] = [self.policy_tree.root]
        self.reasoning_chain: List[BasicNode] = [self.root]

        # Jump into the mcts loop toward updating the policy tree
        while not self.is_node_sink(self.reasoning_chain[-1].identity):
            # Get the current policy and the current thought
            # The current thought derives from the current policy
            cur_policy_node = self.policy_chain[-1]
            cur_thought_node = self.reasoning_chain[-1]

            # Perform the policy exploration of the p-RAR paper
            # Note that this process will be two nodes to the thought structure
            # 1. Added a thought node, either exploration or normal generation
            # 2. Added a policy summarization thought
            policy_thought_id, next_thought_id = self.policy_exploration()

            # Check whether the policy exists in the child of
            # the current policy node
            policy_node_candidates = self.policy_tree.get_successor_nodes(
                cur_policy_node.identity
            )

            is_exist = "false"
            if len(policy_node_candidates) > 0:
                logging.info(
                    "Judging whether the new policy exists in successors of N-%s.",
                    cur_policy_node.identity,
                )
                exist_thoughts, infer_info = self.thought_model.compare_policy(
                    target_policy_thought_node=self.node_pool[policy_thought_id],
                    policy_node_pool=policy_node_candidates,
                )
                is_exist = exist_thoughts[0]
                infer_info = infer_info[0]
                # Add the policy existence comparison to the thought structure
                self.add_node(
                    thought=is_exist,
                    prev_node_id=policy_thought_id,
                    step_idx=self.node_pool[policy_thought_id].step_idx,
                    thought_evaluation=None,
                    thought_inference=infer_info,
                    node_name="PolicyExistenceThought",
                    position="PolicyExistenceIntermediate",
                    step_name="PolicyExistenceComparison",
                    growth="Un-growable",
                    edge_type="PolicyExistenceReasoning",
                )

            if "true" in is_exist.lower():
                logging.info("Selecting best policy as the new policy exists.")
                # Select the best current policy to guide the reasoning
                best_policy_node = self.selection(
                    cur_thought_node=cur_thought_node,
                    cur_policy_node=cur_policy_node,
                )
                # Generate the next thought based on the selected policy
                thoughts, infer_info = self.thought_model.generate_policy_next_thoughts(
                    thought_chain=self.reasoning_chain,
                    policy_chain=self.policy_chain,
                    policy_node=best_policy_node,
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
                # Append the policy
                self.policy_chain.append(best_policy_node)

                # Add the current thought to the policy
                next_thought_node = self.node_pool[next_thought_id]
                value_thoughts, infer_info = self.thought_model.assess_thought_policy(
                    thought_chain=self.reasoning_chain,
                    thought_node=next_thought_node,
                    policy_node=best_policy_node,
                )
                llm_value = float(value_thoughts[0])
                infer_info = infer_info[0]
                self.add_node(
                    thought=llm_value,
                    prev_node_id=next_thought_node.identity,
                    step_idx=next_thought_node.step_idx,
                    thought_evaluation=None,
                    thought_inference=infer_info,
                    node_name="PolicyAssessmentThought",
                    position="PolicyAssessmentIntermediate",
                    step_name="PolicyAssessment",
                    growth="Un-growable",
                    edge_type="PolicyAssessmentReasoning",
                )
                # Extend the node by adding the thought origin
                # and the thought evaluation
                cur_thought_str = str(cur_thought_node.thought)
                if cur_thought_node.step_idx == 0:
                    cur_thought_str = str(cur_thought_node.thought.question)
                self.policy_tree.extend_node(
                    node_id=best_policy_node.identity,
                    thought_origin=cur_thought_str,
                    # Add the [num_wins, v_llm, n_visits]
                    thought_evaluation=[0, llm_value, 1],
                )
            else:
                logging.info("Performing expansion of MCTS:")
                added_policy_node_id = self.expansion(
                    cur_policy_node=cur_policy_node,
                    cur_thought_node=cur_thought_node,
                    policy_thought_node=self.node_pool[policy_thought_id],
                    next_thought_node=self.node_pool[next_thought_id],
                )
                # Add the new policy to the policy chain
                new_policy_node = self.policy_tree.node_pool[added_policy_node_id]
                self.policy_chain.append(new_policy_node)

            # Add the next thought to the reasoning chain
            thought_node = self.node_pool[next_thought_id]
            self.reasoning_chain.append(thought_node)

            # Save the structure to the disk
            # Save the updated policy tree to the same folder
            # of the thought structure
            self.save_structure()
            self.policy_tree.save_structure(
                foldername="policy_tree", location=self.create_save_folder()
            )

            # Save the policy tree to the same folder
            if self.visualizer is not None:
                self.visualizer.visualize(
                    self.graph,
                    self.node_pool,
                    save_name=f"Step_{thought_node.step_idx}",
                )
                self.visualizer.visualize(
                    self.policy_tree.graph,
                    self.policy_tree.node_pool,
                    save_name=f"Policy Step_{self.policy_chain[-1].step_idx}",
                )

            # Jump out of the loop if a new policy is added
            # to the policy tree
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
                self.policy_tree.graph,
                self.policy_tree.node_pool,
                save_name="policy_built_structure",
            )

    def compute_values(self, thought_node: BasicNode, policy_node: PolicyNode):
        """Compute the value of a policy."""
        # A list, each is string of the thought origin
        thought_origins = policy_node.thought_origins
        # A list, each is (num_wins, v_llm)
        thought_evaluations = policy_node.thought_evaluations

        # Get the indexes of the K-nearest thoughts
        top_idx, distances = self.embedder.get_neighbors(
            query=str(thought_node.thought.question),
            documents=thought_origins,
            num_neighbors=self.num_neighbor_thoughts,
        )

        neighbor_evals = [thought_evaluations[idx] for idx in top_idx]
        # Compute the averaged win rate and llm
        # As each thought origin has the format of (num_wins, v_llm, n_visits)
        # we should compute the inner average before computing the outer average
        avg_win = np.mean([score[0] / score[3] for score in neighbor_evals])
        v_llm = np.mean([score[1] / score[3] for score in neighbor_evals])

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
        self.policy_tree = None
        self.reasoning_chain: List[BasicNode] = []
        self.policy_chain: List[PolicyNode] = []
