"""
The prompter to organize the comment prompts for the Boosting of Thoughts (BoT).
"""

from llmpebase.model.prompting.base import BasicSamplePrompt
from llmpebase.model.prompting.prompt_generic import BasicThoughtPromptFormat


class BoTCommentPrompter:
    """
    A thought prompter to organize the thought prompts for the Boosting of Thoughts (BoT).
    """

    system_prompt = "You are an expert AI checker for math answers, dedicated to evaluating the reasoning chain generated towards addressing the mathematical problem. Judge each reasoning step of this reasoning chain by providing detailed analyses on whether the current step is a logical inference of the previous step and whether the reasoning step is beneficial to the correct solution. Provide advice and suggestions for each reasoning step with errors. Provide recommendation or rejection descriptions for each correct reasoning step."

    feedback_prompt_format = BasicThoughtPromptFormat(
        head="Comment on the given reasoning chain by analyzing its effectiveness in solving the question and evaluating the logic and correctness of each step.\n\n{}\nReasoning chain:\n{}\n\n\n",
        content="First, summarize the conclusion and reason about whether the solution derived from the reasoning chain is correct for the given question. If incorrect, please provide an error report and revision advice on each reasoning step. If correct, please analyze the logic and benefit of each step toward problem-solving. Finally, a confidence score is given after 'Confidence score:' for readability. The score ranges from 0.1 to 1 to indicate confidence in the comments.\n",
        target="Generate reasoning chain analysis:\n",
        notice="""""",
        tail="",
        prompt="",
    )

    def organize_chain_feedback_prompt(
        self,
        prompt_sample: BasicSamplePrompt,
        reasoning_chain_prompt: str,
    ):
        """Organize the prompt for the chain feedback."""

        feedback_prompt = BasicThoughtPromptFormat(**self.feedback_prompt_format)
        feedback_prompt.head = feedback_prompt.head.format(
            prompt_sample.question, reasoning_chain_prompt
        )

        return feedback_prompt
