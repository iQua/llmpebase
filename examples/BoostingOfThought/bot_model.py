"""
Implementation of the model used for FoT approach.
"""
import re
from typing import List

from llmpebase.models.LMs_prompting.residual_tree_of_thoughts import ThoughtNode


class ThoughtModel:
    """The class to support the thought generation."""

    prompt_head = "You are a boosting of thoughts, aiming to solve the problem step by step. You should generate a series of intermediate reasoning steps toward addressing a given task described by the user."

    def __init__(self, request_model) -> None:
        self.request_model = request_model

    def organize_thoughs_chain_prompt(self, thoughts_node_chain: List[ThoughtNode]):
        """Organizing thoughts chain into the prompt."""
        # initial prompt should be the thought of the root noe
        intermediate_thoughts_node = thoughts_node_chain[1:]
        intermediate_steps = [
            f"'{thought_node.though}'. 'Evaluate Score: {thought_node.thought_score}'"
            for thought_node in intermediate_thoughts_node
        ]
        intermediate_steps = "\n".join(intermediate_steps)
        reasoning_chain_prompt = f"""{intermediate_steps}"""
        return reasoning_chain_prompt

    def organize_next_though_prompt(self, thoughts_node_chain: List[ThoughtNode]):
        """Generating the prompt for next thought."""
        task_prompt = thoughts_node_chain[0].thought
        chain_prompt = self.organize_thoughs_chain_prompt(thoughts_node_chain)

        prompt = f"""{self.prompt_head} But, instead of showing all steps in one response, you should present only one reasoning step in each response by learning from the previous reasoning steps.\n
        Devise the best possible solution for the task: {task_prompt}. \n
        Below are the previous reasoning steps, presented in order, accompanied by their evaluated scores (A higher score means the reasoning step is more likely to complete the task.): \n
        {chain_prompt}
        By learning from the given previous reasoning steps (you can ignore previous steps when the above space is empty), please include one possible next reasoning step toward solving the task in your response. In each step, you can only select two from the number set to perform Addition, subtraction, multiplication, or division to obtain a new number, which is combined with the remaining number to get a new number set for the next step.
        """

        return prompt

    def organize_though_evaluation_prompt(
        self, thoughts_node_chain: List[ThoughtNode], thought: str
    ):
        """Organizing the prompt for thought evaluation."""

        task_prompt = thoughts_node_chain[0].thought
        chain_prompt = self.organize_thoughs_chain_prompt(thoughts_node_chain)
        prompt = f"""{self.prompt_head}. By analyzing the given task and the previous reasoning steps, you should evaluate the newly generated reasoning step by presenting a score to measure how much this step approaches the final solution. \n
        Devise the best possible solution for the task: {task_prompt}. \n\n
        Below are the reasoning steps, presented in order, accompanied by their evaluated scores: \n
        {chain_prompt}
        Based on these given steps, please evaluate this thought:
        {thought}
        \n\n
        by producing a value ranging from 0 to 1, where 0 means this thought is not related to the final solution and 1 means this thought is the solution. 
        The generated response should include one sub-sentence with the format: 'Evaluation score:' for users to read. 
        """

        return prompt

    def generate_thoughts(
        self, thoughts_node_chain: List[ThoughtNode], num_thoughts: int = 2
    ) -> List[str]:
        """Generating one thought based on the existing thought chain."""
        prompt = self.organize_next_though_prompt(thoughts_node_chain)

        print(prompt)

        responses = self.request_model.perform_request(
            user_prompt=prompt, per_request_responses=num_thoughts
        )
        thoughts = self.request_model.extract_responses_content(responses)
        return thoughts

    def evaluate_though_chain(
        self, thoughts_node_chain: List[ThoughtNode], thought: str
    ):
        """Evaluating the thought chain by LLMs."""
        prompt = self.organize_though_evaluation_prompt(thoughts_node_chain, thought)
        responses = self.request_model.perform_request(
            user_prompt=prompt, per_request_responses=1
        )
        evaluation = self.request_model.extract_responses_content(responses)[0]

        # Extract the evaluation score
        score = 0
        match = re.search(r"Evaluation score[^0-9]*(\d+(\.\d+)?)", evaluation, re.I)
        if match:
            score = float(match.group(1))

        return score

    def measure_similarity(
        self, thought_a, thought_b, priorities: list = None, rejections: list = None
    ):
        """Measuring the similarity between two thoughts."""
        priority = ", ".join(priorities) if priorities is not None else ""
        rejection = (
            ", ".join(rejections) if rejections is not None else "sentence structure"
        )
        prompt = f"""
        Evaluate similarity between two paragraphs, prioritizing {priority} but ignoring {rejection}: \n 
        ###
        First paragraph:
        {thought_a} \n
        Second paragraph:
        {thought_b} \n
        ###
        Show the result by producing a value scale from 0 to 1, where 0 indicates no similarity and 1 indicates identical semantics. \n\n
        The generated response should include one sub-sentence with the format: 'Similarity score:' for users to read. 
        """
        responses = self.request_model.perform_request(
            user_prompt=prompt, per_request_responses=1
        )
        similairity_answer = self.request_model.extract_responses_content(responses)[0]

        # Extract the similairity score
        score = 0
        match = re.search(
            r"Similarity score[^0-9]*(\d+(\.\d+)?)", similairity_answer, re.I
        )
        if match:
            score = float(match.group(1))
        return score
