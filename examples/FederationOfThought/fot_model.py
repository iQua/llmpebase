"""
Implementation of the model used for FoT approach.
"""
import re
from typing import List

from vgbase.models.LMs_prompting.residual_tree_of_thoughts import ThoughtNode
from vgbase.models.LMs.chatgpts import ChatGPTAPIRequest


class ChatGPTModel(ChatGPTAPIRequest):
    """The class to support the model generation."""

    prompt_head = "You are a TreeofThoughts, a superintelligent AI model devoted to helping humans by any means necessary. You aim to generate a series of intermediate reasoning steps toward addressing a given task described by the user"

    def organize_thoughs_chain_prompt(self, thoughts_chain: List[ThoughtNode]):
        """Organizing thoughts chain into the prompt."""
        # initial prompt should be the thought of the root noe
        intermediate_thoughts = thoughts_chain[1:]
        intermediate_steps = [
            f"'{though}'. 'Evaluate Score: {score}'"
            for though, score in intermediate_thoughts
        ]
        intermediate_steps = "\n".join(intermediate_steps)
        reasoning_chain_prompt = f"""
        ###{intermediate_steps}###
        
        """
        return reasoning_chain_prompt

    def organize_next_though_prompt(self, thoughts_chain: List[ThoughtNode]):
        """Generating the prompt for next thought."""
        initial_prompt, _ = thoughts_chain[0]
        chain_prompt = self.organize_thoughs_chain_prompt(thoughts_chain)

        prompt = f"""{self.prompt_head}. By learning from given previous reasoning steps, you should generate one next possible reasoning step at each time to gradually approach the final solution. \n
        Devise the best possible solution for the task: {initial_prompt}. \n\n
        Below are the reasoning steps, presented in order, accompanied by their evaluated scores: \n
        {chain_prompt}
        Based on these given steps, complete the given task by generating only one next reasoning step instead of giving the user all at once. Be simple. Be direct. Provide one intuitive and logical step as soon as you think of it. Thus, please present the user with only one single possible reasoning step and the corresponding solution that is closer to the right solution.
        """

        return prompt

    def organize_though_evaluation_prompt(
        self, thoughts_chain: List[ThoughtNode], thought: str
    ):
        """Organizing the prompt for thought evaluation."""

        initial_prompt, _ = thoughts_chain[0]
        chain_prompt = self.organize_thoughs_chain_prompt(thoughts_chain)
        prompt = f"""{self.prompt_head}. By analyzing the given task and the previous reasoning steps, you should evaluate the newly generated reasoning step by presenting a score to measure how much this step approaches the final solution. \n
        Devise the best possible solution for the task: {initial_prompt}. \n\n
        Below are the reasoning steps, presented in order, accompanied by their evaluated scores: \n
        {chain_prompt}
        Based on these given steps, please evaluate this thought:
        ###
        {thought}
        ### \n\n
        by producing a value ranging from 0 to 1, where 0 means this thought is not related to the final solution and 1 means this thought is the solution. 
        The generated response should include one sub-sentence with the format: 'Evaluation score:' for users to read. 
        """

        return prompt

    def generate_thoughts(
        self, thoughts_chain: List[ThoughtNode], num_thoughts: int = 2
    ) -> List[str]:
        """Generating one thought based on the existing thought chain."""
        prompt = self.organize_next_though_prompt(thoughts_chain)

        responses = self.perform_request(
            user_prompt=prompt, per_request_responses=num_thoughts
        )
        thoughts = self.extract_answer(responses)
        return thoughts

    def evaluate_though_chain(self, thoughts_chain: List[ThoughtNode], thought: str):
        """Evaluating the thought chain by LLMs."""
        prompt = self.organize_though_evaluation_prompt(thoughts_chain, thought)
        responses = self.perform_request(user_prompt=prompt, per_request_responses=1)
        evaluation = self.extract_answer(responses)[0]

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
        responses = self.perform_request(user_prompt=prompt, per_request_responses=1)
        similairity_answer = self.extract_answer(responses)[0]

        # Extract the similairity score
        score = 0
        match = re.search(
            r"Similarity score[^0-9]*(\d+(\.\d+)?)", similairity_answer, re.I
        )
        if match:
            score = float(match.group(1))
        return score
