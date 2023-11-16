"""
Implementation of the reasoner with the experience recall used by BoT.
"""
import re
import time
import random
from typing import List

from llmpebase.models.prompting.tree_thoughts import ThoughtNode


class ExperienceRecallReasoner:
    """A reasoner to support the experience recall of BoT."""

    system_prompt = """You are an expert on mathematical problems. Perform step-by-step reasoning toward problem solving by first learning from an ensemble of trial-and-error reasoning experiences. Such trial-and-error reasoning experience specifically contains error reports and detailed advice on how to revise historical reasoning steps. Always recall these listed experiences before generating a new reasoning step, thereby avoiding making the same mistakes and reusing correct steps to generate better reasoning steps to solve the task."""

    # trial-and-error reasoning experiences
    experiences = ""

    def __init__(self, request_model) -> None:
        self.request_model = request_model

        # A container to store each experience
        self.experience_container = []

    def create_experience(self, feedback: str, chain_content_prompt: str):
        """Create the experience."""

        experience = f"""{chain_content_prompt}\n\n {feedback}"""
        return experience

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

    def organize_though_chain_prompt(self, node_thought_chain: List[ThoughtNode]):
        """Organizing thoughts chain into the prompt."""
        # initial prompt should be the thought of the root noe
        intermediate_thoughts_node = node_thought_chain[1:]

        intermediate_steps = [
            f"""Reasoning Step {idx+1}: {thought_node.thought}. Evaluate Score: {thought_node.thought_score}"""
            for idx, thought_node in enumerate(intermediate_thoughts_node)
        ]
        intermediate_steps = "\n\n\t".join(intermediate_steps)
        reasoning_chain_prompt = f"""{intermediate_steps}"""
        return reasoning_chain_prompt

    def organize_next_thought_prompt(self, node_thought_chain: List[ThoughtNode]):
        """Generating the prompt for next thought."""
        task_prompt = node_thought_chain[0].thought
        chain_prompt = self.organize_though_chain_prompt(node_thought_chain)

        prompt = f"""{task_prompt}. \n First of all, Recall historical reasoning experience: \n\n {self.experiences} \n\n Please make one step of reasoning to generate only one next possible reasoning step. This next reasoning step is the subsequential step from the following ordered previous steps, accompanied by their evaluated scores (A higher score means the reasoning step is more likely to complete the task.): \n\t{chain_prompt}\n\n Based on listed previous reasoning steps (ignore them when the above space is empty), generate one single next possible step following the Task rule. (Emphasize: Please generate only one single next possible reasoning step of the given steps.))
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
        print("-----------------------------------------------------------")
        print(prompt)

        # To ensure the diversity of the generated thoughts
        diverse_scale = 5
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
        print(thoughts)
        print("-----------------------------------------------------------")

        # Wait for 30 seconds to avoid the rate limit of the API
        if self.request_model.has_request_limit():
            time.sleep(40)

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
        similairity_answer = self.request_model.extract_response_contents(responses)[0]

        # Extract the similairity score
        score = 0
        match = re.search(
            r"Similarity score[^0-9]*(\d+(\.\d+)?)", similairity_answer, re.I
        )
        if match:
            score = float(match.group(1))
        return score
