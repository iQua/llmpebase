"""
Implementation of the model used for BoT Model.
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
            f"""Step {idx+1}: '{thought_node.thought}. Evaluate Score: {thought_node.thought_score}'"""
            for idx, thought_node in enumerate(intermediate_thoughts_node)
        ]
        intermediate_steps = "\n\n".join(intermediate_steps)
        reasoning_chain_prompt = f"""{intermediate_steps}"""
        return reasoning_chain_prompt

    def organize_next_thought_prompt(self, thoughts_node_chain: List[ThoughtNode]):
        """Generating the prompt for next thought."""
        task_prompt = thoughts_node_chain[0].thought
        chain_prompt = self.organize_thoughs_chain_prompt(thoughts_node_chain)

        prompt = f"""{self.prompt_head} But, instead of showing all steps in one response, you should present only one reasoning step in each response by learning from the previous reasoning steps.\n
        Devise the best possible solution for the task: {task_prompt}. \n
        Below are the previous reasoning steps, presented in order, accompanied by their evaluated scores (A higher score means the reasoning step is more likely to complete the task.): \n
        {chain_prompt}

        Based on the obtained previous reasoning steps (you can ignore them when the above space is empty), please include one possible next reasoning step toward solving the task in your response.)
        """

        return prompt

    def generate_thoughts(
        self, thoughts_node_chain: List[ThoughtNode], num_thoughts: int = 2
    ) -> List[str]:
        """Generating one thought based on the existing thought chain."""
        prompt = self.organize_next_thought_prompt(thoughts_node_chain)

        print(prompt)

        responses = self.request_model.perform_request(
            user_prompt=prompt, per_request_responses=num_thoughts
        )
        answers = self.request_model.extract_responses_content(responses)
        print(answers)
        thoughts = self.request_model.extract_contents_target_answer(answers)
        print(thoughts)
        return thoughts

    def organize_thought_evaluation_prompt(
        self, thoughts_node_chain: List[ThoughtNode], thought: str
    ):
        """Organizing the prompt for thought evaluation."""

        task_prompt = thoughts_node_chain[0].thought
        chain_prompt = self.organize_thoughs_chain_prompt(thoughts_node_chain)
        prompt = f"""
        Devise the best possible solution for the task: {task_prompt}. \n\n
        Below are the reasoning steps, presented in order, accompanied by their evaluated scores given by you: \n
        {chain_prompt}
        {thought}

        Based on these intermediate reasoning steps, whether the task can be solved based on your evaluation? Please give me an evaluation score ranging from 0 to 1, where a higher score means that the task is more likely solved by the given reasoning step. 
        The generated response should include one sub-sentence with the format: 'Evaluation score:' for users to read. 
        """

        return prompt

    def evaluate_thought_chain(
        self, thoughts_node_chain: List[ThoughtNode], thought: str
    ):
        """Evaluating the thought chain by LLMs."""
        prompt = self.organize_thought_evaluation_prompt(thoughts_node_chain, thought)
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

    def organize_thoughts_chain_feedback_prompt(
        self, thoughts_node_chain: List[ThoughtNode]
    ):
        """Organizing the prompt for thoughts feedback."""

        task_prompt = thoughts_node_chain[0].thought
        chain_prompt = self.organize_thoughs_chain_prompt(thoughts_node_chain)
        prompt = f"""
        For the task: {task_prompt}. \n
        Below are the obtained reasoning steps, presented in order, accompanied by their evaluated scores (A higher score means the reasoning step is more likely to complete the task.) given by you: \n
        {chain_prompt}

        Please evaluate these reasoning steps by analyzing whether they can be performed step-by-step to obtain a correct solution that reaches the target of the task? There are four alternative conclusions: (1) impossible; (2) possible but more subsequent steps are required; (3) possible but should revise some steps; (4) possible as the idea is correct but need to do it again, (4) possible.
        
        If your conclusion is (1), (2), or (3). Please list in detail the reasons for the failure of these steps one by one and also list the corresponding advice on how to fix each of them. 
        """

        return prompt

    def get_thought_chain_feedback(self, thoughts_node_chain: List[ThoughtNode]):
        """Getting the feedback of the thought chain from the LLMs."""
        prompt = self.organize_thoughts_chain_feedback_prompt(thoughts_node_chain)

        print("------------ FeedBack --------")
        print(prompt)
        responses = self.request_model.perform_request(
            user_prompt=prompt, per_request_responses=1
        )
        feedback = self.request_model.extract_responses_content(responses)
        return feedback

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
