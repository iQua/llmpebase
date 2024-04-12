"""
Implementation of prompts for thought structure.
"""

from typing import List

from llmpebase.model.thought_structure import base
from llmpebase.prompt.generic import BasicThoughtPromptFormat
from llmpebase.prompt import (
    BaseSystemPrompts,
    BaseThoughtPrompts,
)
from llmpebase.prompt import format_prompt


class ThoughtStructurePrompter:
    """A base class to organize the prompt of the thought structure."""

    # Flags for the start and end of the thought chain
    thought_start_flag: str = "<Steps>"
    thought_end_flag: str = "<\\Steps>"

    # The head of each step
    step_head: str = "Step {}: "
    # Required System prompts
    generation_system_prompt: str = None
    evaluation_system_prompt: str = None
    similarity_system_prompt: str = None

    # Required prompters
    generation_prompts: BaseThoughtPrompts.generation = None
    evaluation_prompts: BaseThoughtPrompts.evaluation = None
    similarity_prompts: BaseThoughtPrompts.similarity = None

    def __init__(
        self,
        system_prompts: BaseSystemPrompts,
        thought_prompts: BaseThoughtPrompts,
    ):
        """
        Initialize the thought structure prompter.

        It is required to explicitly provide the system prompts and the thought prompts.
        """
        self.system_prompts = system_prompts
        self.generation_system_prompt = system_prompts.generation_prompt
        self.evaluation_system_prompt = system_prompts.evaluation_prompt
        self.similarity_system_prompt = system_prompts.similarity_prompt

        self.thought_prompts = thought_prompts
        self.generation_prompts = thought_prompts.generation
        self.evaluation_prompts = thought_prompts.evaluation
        self.similarity_prompts = thought_prompts.similarity

    def organize_chain_prompt(
        self,
        chain_nodes: List[base.BasicNode],
        with_step_idx: int = False,
        with_flag: int = True,
        with_evaluation_score: bool = True,
        with_indent: int = 0,
    ) -> str:
        """Organize thoughts chain into the prompt.

        :param chain_nodes: A list of thought nodes in the chain.
        :param with_step_idx: Whether to include the step index in the prompt.
        :param with_flag: Whether to include the start and end flag in the prompt.
        """
        # initial prompt should be the thought of the root noe
        intermediate_steps = []

        indent = "" if with_indent == 0 else "\t" * with_indent

        for idx, thought_node in enumerate(chain_nodes):
            step_head = ""
            if with_step_idx:
                step_head = self.step_head.format(idx + 1)
            score = ""
            if thought_node.evaluation_score is not None and with_evaluation_score:
                score = f"Evaluation Score: {thought_node.evaluation_score}"

            intermediate_steps.append(
                f"""{indent}{step_head}{thought_node.thought}\t{score}"""
            )

        intermediate_steps = "\n".join(intermediate_steps)
        reasoning_chain_prompt = f"""{intermediate_steps}"""

        if with_flag:
            reasoning_chain_prompt = f"""{self.thought_start_flag}\n{reasoning_chain_prompt}\n{self.thought_end_flag}"""

        return format_prompt.format_prompt(reasoning_chain_prompt)

    def organize_next_thought_prompt(
        self, chain_nodes: List[base.BasicNode], **kwargs
    ) -> BasicThoughtPromptFormat:
        """Generating the prompt for next thought."""
        root_prompt = str(chain_nodes[0].thought)

        # The chain only contain the first step
        if len(chain_nodes) == 1:
            generation_prompt = BasicThoughtPromptFormat(
                **self.generation_prompts.first_step_prompt
            )
            generation_prompt.head = generation_prompt.head.format(root_prompt)

            return generation_prompt

        chain_prompt = self.organize_chain_prompt(
            chain_nodes[1:], with_flag=True, with_evaluation_score=False
        )

        generation_prompt = BasicThoughtPromptFormat(
            **self.generation_prompts.next_step_prompt
        )
        generation_prompt.head = generation_prompt.head.format(root_prompt)
        generation_prompt.content = generation_prompt.content.format(chain_prompt)
        generation_prompt.target = generation_prompt.target.format(
            self.thought_start_flag, self.thought_end_flag, len(chain_nodes)
        )

        return generation_prompt

    def organize_evaluation_prompt(
        self, thought: str, chain_nodes: List[base.BasicNode]
    ) -> BasicThoughtPromptFormat:
        """Organizing the prompt for thought evaluation."""

        root_prompt = chain_nodes[0].thought
        # Convert the root prompt to be the evaluation prompt
        question = root_prompt.question.content

        if len(chain_nodes) == 1:
            eval_prompt = BasicThoughtPromptFormat(
                **self.evaluation_prompts.first_step_prompt
            )
            eval_prompt.head = eval_prompt.head.format(question)
            eval_prompt.content = eval_prompt.content.format(thought)
            return eval_prompt

        eval_prompt = BasicThoughtPromptFormat(
            **self.evaluation_prompts.current_step_prompt
        )
        chain_prompt = self.organize_chain_prompt(
            chain_nodes[1:], with_flag=True, with_evaluation_score=False
        )
        # +1 here to include the new thought, i.e., the latest reasoning step
        # to evaluate
        step_idx = len(chain_nodes[1:]) + 1

        eval_prompt.head = eval_prompt.head.format(step_idx, question)
        eval_prompt.content = eval_prompt.content.format(
            chain_prompt, step_idx, thought
        )

        return eval_prompt

    def organize_similarity_prompt(
        self, thought_a: str, thought_b: str, chain_nodes: List[base.BasicNode]
    ):
        """Organize the prompt for measuring the similarity between two thoughts."""
        root_prompt = chain_nodes[0].thought
        question = root_prompt.question.content
        sim_prompt = BasicThoughtPromptFormat(**self.similarity_prompts.sim_prompt)
        sim_prompt.head = sim_prompt.head.format(question)
        sim_prompt.content = sim_prompt.content.format(thought_a, thought_b)

        return sim_prompt
