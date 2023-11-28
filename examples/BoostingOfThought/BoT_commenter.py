"""
Implementation of commenter to evaluate the thought chain and provide feedback.
"""


class ReasoningChainCommenter:
    """A commenter to support the feedback generation of BoT."""

    system_prompt = "You are an expert AI checker for math answers, dedicated to evaluating the reasoning chain generated towards addressing the mathematical problem. Judge each reasoning step of this reasoning chain by providing detailed analyses on whether the current step is a logical inference of the previous step and whether the reasoning step is beneficial to the correct solution. Provide advice and suggestions for each reasoning step with errors. Provide recommendation or rejection descriptions for each correct reasoning step."

    chain_feedback_format: str = "  Can this reasoning chain complete the task and reach the target correctly by executing its reasoning steps? why? Write a analysis report with conclusion under 'Anlysis Report:'."

    step_feedback_format: str = "  For each reasoning step, please provide a detailed analysis of whether the current step is a logical inference of the previous step and whether the reasoning step is beneficial to the correct solution. For each reasoning step with errors, please provide an error report and the corresponding advice on revision. For each reasoning step, please provide recommendation or rejection descriptions. Comments should be brief, avoid repeating the same analysis in different steps and follow the format: Reasoning step <idx>. \n Analysis report: .\n Advice: .\n Recommendation or Reject description: . \n"

    confidence_feedback_format: str = "  What is your confidence score on these your evaluations and comments? Please select one value from [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]. The score should be placed after 'Confidence score:' for users to read."

    Notice: str = """Do NOT repeat the listed reasoning step within "------------" in the feedback but only show step index."""

    def __init__(self, request_model) -> None:
        self.request_model = request_model

    def organize_chain_content_prompt(self, reasoning_chain_content: str):
        """Organize the prompt for the content of the chain."""
        chain_content_prompt = f"""Below is a reasoning chain containing reasoning steps presented in order:\n{reasoning_chain_content}"""
        return chain_content_prompt

    def organize_chain_feedback_prompt(
        self, task_prompt: str, chain_content_prompt: str
    ):
        """Organize the prompt for thoughts feedback."""

        prompt = f"""Given task:{task_prompt}.\n{chain_content_prompt}\n\n Please evaluate this reasoning chain by giving detailed comments containing the following content.\n 1.{self.chain_feedback_format}. 2.{self.step_feedback_format}. 3.{self.confidence_feedback_format}.\n\n Notice: {self.Notice}.
        """

        return prompt

    def get_thought_chain_feedback(
        self, task_prompt: str, reasoning_chain_content: str
    ):
        """Get the feedback of the thought chain from the LLMs."""
        chain_content_prompt = self.organize_chain_content_prompt(
            reasoning_chain_content
        )
        prompt = self.organize_chain_feedback_prompt(task_prompt, chain_content_prompt)

        # Forward the generation model to get responses
        responses = self.request_model.perform_request(
            user_prompt=prompt, per_request_responses=3, sys_prompt=self.system_prompt
        )
        response_contents = self.request_model.extract_response_contents(responses)
        # Extract the longest response as the feedback
        feedback = max(response_contents, key=len)
        return feedback, chain_content_prompt
