"""
Testing the tree building approaches.
"""
import random
import re

from typing import List

from llmpebase.models.prompting.residual_tree_of_thoughts import (
    ThoughtNode,
    RToTLevelWise,
    RToTLevelWiseBest,
    RToTLeafWise,
)

from vgbase.config import Config


class TestModel:
    """A simple model to test the tree thoughts."""

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
        ### {intermediate_steps} ###
        
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
        Based on these given steps, complete the given task by generating only one next reasoning step instead of giving the user all at once. Be simple. Be direct. Provide one intuitive and logical step as soon as you think of it. Thus, please present the user with only one single possible reasoning step and the corresponding solution, which is expected to have a higher evaluation score - closer to the right solution.
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
        The generated answer should include one sub-sentence with the format: 'Evaluation score:' for users to read. 
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
        The generated answer should include one sub-sentence with the format: 'Similarity score:' for users to read. 
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

    def perform_request(self, user_prompt: str, per_request_responses: int):
        """Performing the request."""
        subjects = [
            "The cat",
            "A boy",
            "An apple",
            "The sun",
            "The teacher",
            "A robot",
            "The car",
            "The bird",
            "A computer",
            "The river",
            "A tree",
            "The fish",
            "A dog",
            "The moon",
            "A shoe",
        ]

        verbs = [
            "jumped",
            "ran",
            "shined",
            "slept",
            "swam",
            "danced",
            "drove",
            "whispered",
            "screamed",
            "walked",
            "sat",
            "flew",
            "ate",
            "looked",
            "wrote",
        ]

        objects = [
            "over the moon",
            "under the bridge",
            "brightly in the sky",
            "on the grass",
            "in the rain",
            "through the forest",
            "during the storm",
            "on the hill",
            "under the sun",
            "beside the lake",
            "in the room",
            "through the window",
            "without a care",
            "with joy",
            "in the dark",
        ]

        responses = []
        for _ in range(per_request_responses):
            subject = random.choice(subjects[:3])
            verb = random.choice(verbs[:3])
            obj = random.choice(objects[:3])
            eval_score = round(random.random(), 2)
            sim_score = round(random.random(), 2)
            sentence = f"{subject} {verb} {obj}. Evaluation score: {eval_score}, Similarity score: {sim_score}."
            responses.append(sentence)

        return responses

    def extract_answer(self, responses):
        return responses


def _main():
    """The core function for model running."""

    test_model = TestModel()

    model_config = Config().model
    model_config = Config.items_to_dict(model_config._asdict())
    # tree = RToTLevelWise(
    #     model=test_model, n_child_nodes=2, **model_config["tree_settings"]
    # )
    # tree = RToTLevelWiseBest(
    #     model=test_model, n_child_nodes=2, **model_config["tree_settings"]
    # )

    tree = RToTLeafWise(
        model=test_model, n_child_nodes=2, **model_config["tree_settings"]
    )

    tree.construct_tree_root(
        though="This is an interesting test for the proposed approach"
    )

    tree.build_thought_tree()
    tree.print_tree_structure()

    tree.save_tree_to_json(file_name="test.json", save_dir=".")
    tree.save_tree_to_picture(file_name="test1.png", save_dir=".")


if __name__ == "__main__":
    _main()
