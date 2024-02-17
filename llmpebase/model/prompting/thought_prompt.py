"""
Implementation of prompts for thought structure.
This thought structure is typically used during building the thought structure.
"""

from typing import List

from llmpebase.model.thought_structure import base
from llmpebase.model.prompting.prompt_generic import BasicThoughtPromptFormat


class ThoughtStructurePrompt:
    """A base class to organize the prompt of the thought structure."""

    thought_flag: str = "-" * 20
    evaluation_score_flag: str = "Evaluation Score: "
    similarity_score_flag: str = "Similarity Score: "

    step_head: str = "Reasoning Step {}: "

    generation_system_prompt: str = (
        """You are an expert in solving mathematical problems using methodical, step-by-step reasoning. You should solve each question by generating a series of logical reasoning steps, with each response contributing one step in the sequence. Start by reviewing the given problem and any reasoning steps already taken, and then proceed to provide the next logical step in the solution process. You are responsible for carefully crafting each step to construct a clear, logical progression that leads to the solution."""
    )
    evaluation_system_prompt: str = (
        """Your expertise lies in critically evaluating the latest step in a reasoning process toward problem solving. Your analysis focuses on assessing the latest step's validity, logical coherence, and progression after considering a series of reasoning steps. Your role involves identifying any logical fallacies or weaknesses in the latest step. Please conclude the verification with an evaluation score ranging from 0 to 1, in which the higher value means the better reasoning step."""
    )
    similarity_system_prompt: str = (
        """You are an expert at measuring the similarity between two reasoning steps by comparing their mathematical logic, mathematical content overlap, mathematical conclusion, and contribution to the final solution for the question. Output the evaluation result as a score ranging from 0 to 1 with higher value measures better."""
    )

    first_step_generation_prompt = BasicThoughtPromptFormat(
        head="{}Let's focus on carefully generating the first reasoning step.\n",
        content="",
        target="Generate a small and well-crafted first step, i.e., step 1, containing analysis and the corresponding mathematical expression as the start of reasoning.",
        notice="",
        tail="",
        prompt="",
    )

    generation_prompt = BasicThoughtPromptFormat(
        head="{}Let's focus on carefully generating the next possible reasoning step for reasoning steps below.\n",
        content="\n{}\n\n",
        target="For reasoning steps within {}, please generate their best next step containing analysis and the corresponding mathematical expression.",
        notice="",
        tail="",
        prompt="",
    )

    first_evaluation_prompt = BasicThoughtPromptFormat(
        head="Evaluate the first reasoning step for the given question:\n{}\n\n",
        content="The latest reasoning step is: \n{}\n\n",
        target="Evaluate this first step by assessing the validity, logical coherence, and progression and identifying any logical fallacies or weaknesses. Conclude the verification with a score ranging from 0 to 1, where a higher value means a better reasoning step, while 0 represents an critical error.\n",
        notice=f"Present the score after '{evaluation_score_flag}' for readability.\n",
        tail="",
        prompt="",
    )

    evaluation_prompt = BasicThoughtPromptFormat(
        head="Evaluate the latest reasoning step for the given question:\n{}\n\n",
        content="Toward addressing the question, we have, thus far, obtained a series of reasoning steps: \n {}\n\n The latest reasoning step is: \n{}\n\n",
        target="Evaluate the latest step by assessing the validity, logical coherence, and progression and identifying any logical fallacies or weaknesses. Conclude the verification with a score ranging from 0 to 1, where a higher value means a better reasoning step, while 0 represents an critical error.\n",
        notice=f"Present the score after '{evaluation_score_flag}' for readability.\n",
        tail="",
        prompt="",
    )

    sim_prompt = BasicThoughtPromptFormat(
        head="Evaluate the similarity between two reasoning steps generated for addressing the given question: \n{}\n\n",
        content="Below are two reasoning steps to be compared:\n\nA. {}\n\nB. {}\n\n",
        target="Score similarity by measuring their consistency in logic, words, and results. The similarity score ranges from 0 to 1, where a higher score means higher similarity.\n ",
        notice=f"Present the score after '{similarity_score_flag}' for readability.\n",
        tail="",
        prompt="",
    )

    def organize_chain_prompt(
        self,
        chain_nodes: List[base.BasicNode],
        with_step_idx: int = False,
        with_flag: int = True,
        with_evaluation_score: bool = True,
    ) -> str:
        """Organize thoughts chain into the prompt.

        :param chain_nodes: A list of thought nodes in the chain.
        :param with_step_idx: Whether to include the step index in the prompt.
        :param with_flag: Whether to include the start and end flag in the prompt.
        """
        # initial prompt should be the thought of the root noe
        intermediate_steps = []

        for idx, thought_node in enumerate(chain_nodes):
            step_head = ""
            if with_step_idx:
                step_head = self.step_head.format(idx + 1)
            score = ""
            if thought_node.evaluation_score is not None and with_evaluation_score:
                score = f"Evaluation Score: {thought_node.evaluation_score}"

            intermediate_steps.append(
                f"""\t{step_head}{thought_node.thought} {score}"""
            )

        intermediate_steps = "\n".join(intermediate_steps)
        reasoning_chain_prompt = f"""{intermediate_steps}"""

        if with_flag:
            reasoning_chain_prompt = f"""{self.thought_flag}\n{reasoning_chain_prompt}\n{self.thought_flag}"""

        return reasoning_chain_prompt

    def organize_next_thought_prompt(
        self, chain_nodes: List[base.BasicNode], **kwargs
    ) -> BasicThoughtPromptFormat:
        """Generating the prompt for next thought."""
        root_prompt = str(chain_nodes[0].thought)

        # The chain only contain the first step
        if len(chain_nodes) == 1:
            generation_prompt = BasicThoughtPromptFormat(
                **self.first_step_generation_prompt
            )
            generation_prompt.head = generation_prompt.head.format(root_prompt)

            return generation_prompt

        chain_prompt = self.organize_chain_prompt(
            chain_nodes[1:], with_flag=True, with_evaluation_score=False
        )

        generation_prompt = BasicThoughtPromptFormat(**self.generation_prompt)
        generation_prompt.head = generation_prompt.head.format(root_prompt)
        generation_prompt.content = generation_prompt.content.format(chain_prompt)
        generation_prompt.target = generation_prompt.target.format(self.thought_flag)

        return generation_prompt

    def organize_evaluation_prompt(
        self, thought: str, chain_nodes: List[base.BasicNode]
    ) -> BasicThoughtPromptFormat:
        """Organizing the prompt for thought evaluation."""

        root_prompt = chain_nodes[0].thought
        # Convert the root prompt to be the evaluation prompt
        question = root_prompt.question.content

        if len(chain_nodes) == 1:
            eval_prompt = BasicThoughtPromptFormat(**self.first_evaluation_prompt)
            eval_prompt.head = eval_prompt.head.format(question)
            eval_prompt.content = eval_prompt.content.format(thought)
            return eval_prompt

        eval_prompt = BasicThoughtPromptFormat(**self.evaluation_prompt)
        chain_prompt = self.organize_chain_prompt(
            chain_nodes[1:], with_flag=True, with_evaluation_score=False
        )

        eval_prompt.head = eval_prompt.head.format(question)
        eval_prompt.content = eval_prompt.content.format(chain_prompt, thought)

        return eval_prompt

    def organize_similarity_prompt(
        self, thought_a: str, thought_b: str, chain_nodes: List[base.BasicNode]
    ):
        """Organize the prompt for measuring the similarity between two thoughts."""
        root_prompt = chain_nodes[0].thought
        question = root_prompt.question.content
        sim_prompt = BasicThoughtPromptFormat(**self.sim_prompt)
        sim_prompt.head = sim_prompt.head.format(question)
        sim_prompt.content = sim_prompt.content.format(thought_a, thought_b)

        return sim_prompt
