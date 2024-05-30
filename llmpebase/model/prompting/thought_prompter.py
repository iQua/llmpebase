"""
Implementation of prompts for thought structure.
"""

import re
from typing import List

from llmpebase.model.thought_structure import base
from llmpebase.prompt.generic import BasicThoughtPromptFormat
from llmpebase.prompt import (
    BaseSystemPrompts,
    BaseThoughtPrompts,
)
from llmpebase.prompt import format_prompt


def match_step_head(step_head, thought):
    """Match the step head in the thought."""
    step_head = step_head.replace(".", "")
    step_head = step_head.strip()

    # Escape special characters in step_head and compile a case-insensitive regex pattern
    pattern = re.compile(re.escape(step_head), re.IGNORECASE)

    # Search for the pattern in the thought
    match = pattern.search(thought)

    # Return True if a match is found, otherwise False
    return match is not None


class ThoughtStructurePrompter:
    """A base class to organize the prompt of the thought structure."""

    # Flags for the start and end of the thought chain
    thought_chain_start_flag: str = "<Step Chain>"
    thought_chain_end_flag: str = "<\\Step Chain>"

    # The head of each step
    step_head: str = "Step {}. "

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

    def organize_node_block_prompt(
        self,
        nodes: List[base.BasicNode],
        content_attr: str,
        head_format: str,
        start_flag: str = None,
        end_flag: str = None,
        with_index: bool = False,
        with_indent: int = 0,
    ):
        """
        Organize the block of nodes in the prompt.

        This is used as a general purpose block prompt organize function.
        """
        # initial prompt should be the thought of the root noe
        indent = "" if with_indent == 0 else "\t" * with_indent
        block_content = []
        for idx, node in enumerate(nodes):
            item_head = "" if head_format is None else head_format
            if head_format is not "":
                item_head = item_head.format(node.step_idx)
            if with_index:
                item_head = f"""({idx+1}). {item_head}"""

            node_content = getattr(node, content_attr)
            block_content.append(f"""{indent}{item_head} {node_content}""")

        block_content = "\n".join(block_content)
        block_content = f"""{block_content}"""

        if start_flag is not None:
            block_content = f"""{start_flag}\n{block_content}\n{end_flag}"""

        return format_prompt.format_prompt(block_content)

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
        :param is_adaptive_step_idx: Whether include the step idx adaptively,
         i.e.,
        """
        # initial prompt should be the thought of the root noe
        intermediate_steps = []

        indent = "" if with_indent == 0 else "\t" * with_indent

        for idx, thought_node in enumerate(chain_nodes):
            step_head = ""
            thought = thought_node.thought
            if with_step_idx:
                step_head = self.step_head.format(idx + 1)
                if match_step_head(step_head, thought):
                    step_head = ""

            score = ""
            if thought_node.evaluation_score is not None and with_evaluation_score:
                score = f"Evaluation Score: {thought_node.evaluation_score}"

            intermediate_steps.append(f"""{indent}{step_head}{thought}\t{score}""")

        intermediate_steps = "\n".join(intermediate_steps)
        reasoning_chain_prompt = f"""{intermediate_steps}"""

        if with_flag:
            reasoning_chain_prompt = f"""{self.thought_chain_start_flag}\n{reasoning_chain_prompt}\n{self.thought_chain_end_flag}"""

        return format_prompt.format_prompt(reasoning_chain_prompt)

    def organize_next_thought_prompt(
        self, chain_nodes: List[base.BasicNode], **kwargs
    ) -> BasicThoughtPromptFormat:
        """Generating the prompt for next thought."""
        root_prompt = str(chain_nodes[0].thought)

        # The chain only contain the root, i.e., question.
        if len(chain_nodes) == 1:
            generation_prompt = BasicThoughtPromptFormat(
                **self.generation_prompts.first_step_prompt
            )
            generation_prompt.head = generation_prompt.head.format(root_prompt)

            return generation_prompt

        chain_prompt = self.organize_chain_prompt(
            chain_nodes[1:],
            with_step_idx=True,
            with_flag=True,
            with_evaluation_score=False,
        )

        generation_prompt = BasicThoughtPromptFormat(
            **self.generation_prompts.next_step_prompt
        )
        generation_prompt.head = generation_prompt.head.format(root_prompt)
        generation_prompt.content = generation_prompt.content.format(chain_prompt)
        generation_prompt.target = generation_prompt.target.format(
            self.thought_chain_start_flag, len(chain_nodes)
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
            chain_nodes[1:],
            with_step_idx=True,
            with_flag=True,
            with_evaluation_score=False,
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
