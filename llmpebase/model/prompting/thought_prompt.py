"""
Implementation of prompts for thought structure.
This thought structure is typically used during building the thought structure.
"""

from typing import List

from llmpebase.model.thought_structure import base


class ThoughtStructurePrompt:
    """A base class to organize the prompt of the thought structure."""

    thought_flag: str = "-----------------------------------"
    step_head: str = "Reasoning Step {}: "

    generation_system_prompt: str = """You are an expert at solving mathematical problems by performing step-by-step reasoning with each response containing only one reasoning step at a time. Each time, you generate one step as the subsequent reasoning step of the given reasoning steps so that the new reasoning chain approaches the solution of the given question."""
    generation_head: str = "{} Let's focus on generating one step at a time."
    generation_content: str = "Below is an obtained reasoning chain containing reasoning steps presented in order:\n{}"
    generation_target: str = """Based on above reasoning steps within '{}', please generate one subsequent possible reasoning step. Please only provide one single step next to the given reasoning steps. When there are no given steps, please generate the first reasoning step."""
    generation_answer: str = """Generate one next reasoning step."""

    evaluation_system_prompt: str = """You are an expert at evaluating a newly generated reasoning step, the next step of a series of given reasoning steps for the question. Your evaluation depends on the condition after including the new reasoning step into the given steps, whether the reasoning chain is logically correct, and approach the final solution. Output the evaluation result as a score ranging from 0 to 1 with higher value measures better."""
    evaluate_head: str = "Evaluate the reasoning step for the given question."
    evaluate_content: str = "We have, thus far, obtained a reasoning chain containing reasoning steps and their respective evaluation scores below:\n{}\n\n For the above reasoning steps within {}, we obtain a new next reasoning step: {}\n\n"
    evaluate_metric: str = "Score this next step by measuring its logic flow, correctness, and benefit to reach or approach a final solution for the given question. The evaluate score ranges from 0 to 1, where a higher score means better reasoning steps.\n"
    evaluate_notice: str = "Only output the score itself.\n"
    evaluate_answer: str = "Evaluate score:"

    similarity_system_prompt: str = """You are an expert at measuring the similarity between two reasoning steps by comparing their mathematical logic, mathematical content overlap, mathematical conclusion, and contribution to the final solution for the question. Output the evaluation result as a score ranging from 0 to 1 with higher value measures better."""
    sim_head: str = "Evaluate the similarity between two reasoning steps generated for addressing the given question.\n"
    sim_question: str = "\n{}\n"
    sim_content: str = (
        "Below are two reasoning steps to be compared.\nA. {}\n\nB. {}\n\n"
    )
    sim_metric: str = "Score similarity by measuring their consistency in logic, words, and results. The similarity score ranges from 0 to 1, where a higher score means higher similarity.\n "
    sim_notice: str = "Only output the score itself.\n"
    sim_answer: str = "Similarity score:"

    def organize_chain_prompt(
        self,
        chain_nodes: List[base.BasicNode],
        with_step_idx: int = False,
        with_flag: int = True,
        with_evaluation_score: bool = True,
    ):
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
            if thought_node.thought_score is not None and with_evaluation_score:
                score = f"Evaluation Score: {thought_node.thought_score}"

            intermediate_steps.append(
                f"""\t{step_head}{thought_node.thought}. {score}"""
            )

        intermediate_steps = "\n".join(intermediate_steps)
        reasoning_chain_prompt = f"""{intermediate_steps}"""

        if with_flag:
            reasoning_chain_prompt = f"""{self.thought_flag}\n{reasoning_chain_prompt}\n{self.thought_flag}"""

        return reasoning_chain_prompt

    def organize_next_thought_prompt(self, chain_nodes: List[base.BasicNode]):
        """Generating the prompt for next thought."""
        root_prompt = str(chain_nodes[0].thought)
        chain_prompt = self.organize_chain_prompt(chain_nodes[1:], with_flag=True)
        head_prompt = self.generation_head.format(root_prompt)
        generation_content = self.generation_content.format(chain_prompt)
        generation_target = self.generation_target.format(self.thought_flag)
        prompt = f"""{head_prompt} {generation_content}\n\n{generation_target}\n{self.generation_answer}
        """
        return prompt

    def organize_evaluation_prompt(
        self, thought: str, chain_nodes: List[base.BasicNode]
    ):
        """Organizing the prompt for thought evaluation."""

        root_prompt = chain_nodes[0].thought
        # Convert the root prompt to be the evaluation prompt
        question = root_prompt.question.content

        chain_prompt = self.organize_chain_prompt(chain_nodes[1:], with_flag=True)

        content = self.evaluate_content.format(chain_prompt, self.thought_flag, thought)
        prompt = f"""{self.evaluate_head}\n{question}\n{content}{self.evaluate_metric}{self.evaluate_notice}{self.evaluate_answer}"""

        return prompt

    def organize_similarity_prompt(
        self, thought_a: str, thought_b: str, chain_nodes: List[base.BasicNode]
    ):
        """Organize the prompt for measuring the similarity between two thoughts."""
        root_prompt = chain_nodes[0].thought
        question = root_prompt.question.content
        sim_question = self.sim_question.format(question)
        content = self.sim_content.format(thought_a, thought_b)
        return f"""{self.sim_head}{sim_question}{content}{self.sim_metric}{self.sim_notice}{self.sim_answer}"""
