"""
Implementation of the reasoner with the experience recall used by BoT.
"""
import re
import time
import random
from typing import List

from llmpebase.model.prompting.tree_thoughts import ThoughtNode


class ExperienceRecallReasoner:
    """A reasoner to support the experience recall of BoT."""

    system_prompt = """You are an expert on mathematical problems. Perform step-by-step reasoning toward problem solving by first learning from an ensemble of trial-and-error reasoning experiences. Such trial-and-error reasoning experience specifically contains error reports and detailed advice on how to revise historical reasoning steps. Always recall these listed experiences before generating a new reasoning step, thereby avoiding making the same mistakes and reusing correct steps to generate better reasoning steps to solve the task."""

    # trial-and-error reasoning experiences
    experiences = ""

    def __init__(self, request_model) -> None:
        self.request_model = request_model

        # A container to store each experience
        self.experience_container = []

        # Organize the experiences in the container
        self.organize_container_experiences()

    def memory_experience(self, feedback: str):
        """
        Collect the experience from the feedback, which contains error reports
        and detailed advice on how to revise previously generated reasoning steps.
        """
        if len(feedback.strip()) != 0:
            self.experience_container.append(feedback)

        self.organize_container_experiences()

    def organize_container_experiences(self):
        """Organize experiences in the container into a single string."""
        self.experiences = ""
        for i, reason in enumerate(self.experience_container, start=1):
            self.experiences += f"######### The {i}-th Reasoning Chain with Comments #########\n{reason}\n\n"

        stop_line = "##################################################################"
        self.experiences = f"""{self.experiences}\n {stop_line}"""

    def organize_next_thought_prompt(self, node_thought_chain: List[ThoughtNode]):
        """Generating the prompt for next thought."""
        task_prompt = node_thought_chain[0].thought
        chain_prompt = self.organize_though_chain_prompt(node_thought_chain)

        prompt = f"""{task_prompt}. Please split the problem into several problems and solve them step by step.\n \n Recall historical reasoning experience (Ignore when experience is empty): \n\n {self.experiences} \n Pay attention to analysis and advice in the above experience to avoid making similar mistakes by following the advice. \n\n Below is a list of ordered reasoning steps, accompanied by their evaluated scores (A higher score means the reasoning step is more likely to complete the task.): \n\t{chain_prompt}\n\nBased on listed reasoning steps only within the above "------------" (i.e., Not the ones in the experience block), please make one step of reasoning to generate only one subsequential possible reasoning step. Just ignore them when there is no listed steps. Notice: Do NOT mistakenly generate the reasoning steps next to the ones in experience, i.e., when the listed reasoning steps within "------------" is empty, you should generate step 1!
        """
        return prompt

    def generate_thoughts(
        self, node_thought_chain: List[ThoughtNode], num_thoughts: int = 2, **kwargs
    ) -> List[str]:
        """Generating one thought based on the existing thought chain.

        This function contains an important observation:
         Given a problem, such as the 1, 1, 4, 6 of the GameOf24 task, the generation model
         will always produce the similar responses. In most cases, 1, 1 will be selcted to
         perform 1 + 1 in the first step.
         Therefore, to ensure the diversity of the generated thoughts, we can generate a massive
         of responses and select two different ones from them.

         Such an observation further verify the importance of experiences because once the model
         can recall experiences to avoid invalid reasoning steps before resposeing, it will
         1. generate multiple different resoning steps, instead of trapping in the similar ones.
         2. generate more correct reasoning steps by learning from experiences.
        """

        prompt = self.organize_next_thought_prompt(node_thought_chain)
        # print("-----------------------------------------------------------")
        # print(prompt)

        # To ensure the diversity of the generated thoughts
        diverse_scale = 1
        num_diverse_thoughts = min(num_thoughts * diverse_scale, 20)

        responses = self.request_model.perform_request(
            user_prompt=prompt,
            per_request_responses=num_diverse_thoughts,
            sys_prompt=self.system_prompt,
        )
        answers = self.request_model.extract_response_contents(responses)
        answers = list(set(answers))
        if len(answers) < num_thoughts:
            # Extend the answers
            answers = answers * max(1, num_thoughts // len(answers))

        thoughts = random.sample(answers, num_thoughts)

        # Wait for 30 seconds to avoid the rate limit of the API
        # if self.request_model.has_request_limit():
        #     time.sleep(30)
        # # print(thoughts)
        return thoughts

    def organize_chain_evaluation_prompt(
        self, node_thought_chain: List[ThoughtNode], thought: str
    ):
        """Organizing the prompt for thought evaluation."""

        task_prompt = node_thought_chain[0].thought
        chain_prompt = self.organize_though_chain_prompt(node_thought_chain)
        prompt = f"""{task_prompt}. \n\n Below are the generated reasoning steps, presented in order, accompanied by their evaluated scores (A higher score means the reasoning step is more likely to complete the task.):\n{chain_prompt}\n{thought}\n\nWhat is your evaluation score for the logic, correctness, and benefit to reaching a final solution for these reasoning steps? Please select one value from [0.1, 0.3, 0.5, 0.7, 0.9, 1.0] as the score, where a higher score means better reasoning steps. The score should be placed after 'Evaluation score:' for users to read.".
        """

        return prompt

    def evaluate_reasoning_thought(
        self, node_thought_chain: List[ThoughtNode], thought: str
    ):
        """Evaluate the quality reasoning thought in the chain by LLMs."""
        prompt = self.organize_chain_evaluation_prompt(node_thought_chain, thought)
        responses = self.request_model.perform_request(
            user_prompt=prompt, per_request_responses=1
        )
        evaluation = self.request_model.extract_response_contents(responses)[0]

        # Extract the evaluation score
        score = 0
        match = re.search(r"Evaluation score[^0-9]*(\d+(\.\d+)?)", evaluation, re.I)
        if match:
            score = float(match.group(1))

        return score

    def measure_similarity(self, thought_a, thought_b):
        """Measuring the similarity between two thoughts."""
        prompt = f"""
        Evaluate the similarity between two paragraphs by showing the percentage of the same words and mathematical numbers: \n 
        ### \nFirst paragraph:{thought_a} \n\nSecond paragraph:\n{thought_b} \n\n###\n\nThe generated response should only contain the computed percentage with the format: 'Similarity score:' for users to read. 
        """
        responses = self.request_model.perform_request(
            user_prompt=prompt, per_request_responses=1
        )
        similairity_answer = self.request_model.extract_response_contents(responses)[0]

        # Extract the similairity score
        score = 0
        match = re.search(
            r"Similarity score[^0-9]*(\d+(\.\d+)?)", similairity_answer, re.I
        )
        if match:
            score = float(match.group(1))
        return score

    @staticmethod
    def organize_though_chain_prompt(
        node_thought_chain: List[ThoughtNode], with_step_idx=False, with_start_end=True
    ):
        """Organizing thoughts chain into the prompt."""
        # initial prompt should be the thought of the root noe
        intermediate_thoughts_node = node_thought_chain[1:]

        intermediate_steps = []

        for idx, thought_node in enumerate(intermediate_thoughts_node):
            prefix_str = ""
            if with_step_idx:
                prefix_str = f"Reasoning Step {idx+1}: "

            intermediate_steps.append(
                f"""\t{prefix_str}{thought_node.thought}. Evaluate Score: {thought_node.thought_score}"""
            )

        intermediate_steps = "\n".join(intermediate_steps)
        reasoning_chain_prompt = f"""{intermediate_steps}\n"""

        start_format = "-----------------------------------"
        end_format = "-----------------------------------"
        if with_start_end:
            reasoning_chain_prompt = (
                f"""{start_format}\n{reasoning_chain_prompt}\n{end_format}"""
            )

        return reasoning_chain_prompt

    @staticmethod
    def create_experience(feedback: str, chain_content_prompt: str):
        """Create the experience."""

        experience = f"""{chain_content_prompt}\n\n {feedback}"""
        return experience
