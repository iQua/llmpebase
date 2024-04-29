"""
The thought prompter for the Boosting of Thoughts (BoT).
"""

from typing import List, Tuple

from llmpebase.model.prompting.base import BasicSamplePrompt, BasicPromptFormat
from llmpebase.model.prompting.thought_prompter import ThoughtStructurePrompter
from llmpebase.prompt import format_prompt


class BoTThoughtPrompter(ThoughtStructurePrompter):
    """
    A thought prompter to organize the thought prompts for the Boosting of Thoughts (BoT).
    """

    experience_start_flag: str = """<Experiences>"""
    experience_end_flag: str = """<\\Experiences>"""

    chain_start_flag: str = """<Chain>"""
    chain_end_flag: str = """<\\Chain>"""

    feedback_start_flag: str = """<Feedback>"""
    feedback_end_flag: str = """<\\Feedback>"""

    experience_system_prompt: str = (
        """Between <Experiences> and <\\Experiences>, you are given trial-and-error reasoning experiences containing reasoning trials and their analysis. Before answering, learn from the given experiences to avoid making the same mistakes, follow correct reasoning, and polish the reasoning steps to generate better ones to solve the task."""
    )

    experience_prompt_format = BasicPromptFormat(
        head="Recall historical reasoning experience (Ignore when experience is empty):\n",
        content="<Experience-{}>\n{}\n<\\Experience-{}>",
        notice="",
        tail="Before generating reasoning steps to answer the question, consider the analysis and advice in the above experience to avoid making similar mistakes and produce better steps.\n\n",
        prompt="",
    )

    def __init__(
        self,
        system_prompts=None,
        thought_prompts=None,
    ):
        super().__init__(
            system_prompts,
            thought_prompts,
        )

        # Adding the experience system prompt to generation_system_prompt
        self.generation_system_prompt = (
            self.generation_system_prompt + "\n" + self.experience_system_prompt
        )

    def organize_experience_prompt(
        self, experiences: List[Tuple[str, str]], with_flag: bool = True
    ):
        """Organize experiences in the container into a single string."""
        experience_prompt = BasicPromptFormat(**self.experience_prompt_format)

        experience_block = []
        for i, experience in enumerate(experiences, start=1):
            solution_str = experience[0]
            feedback = experience[1]

            experience_str = f"""{self.chain_start_flag}\n{solution_str}\n{self.chain_end_flag}\n\n{self.feedback_start_flag}\n{feedback}\n{self.feedback_end_flag}"""

            experience_block.append(
                self.experience_prompt_format.content.format(i, experience_str, i)
            )
        experience_prompt.content = "\n\n".join(experience_block)
        if with_flag:
            experience_prompt.content = f"""{self.experience_start_flag}\n{experience_prompt.content}\n{self.experience_end_flag}\n\n"""

        # Format the prompt
        experience_prompt.content = format_prompt.format_prompt(
            experience_prompt.content
        )

        return experience_prompt

    def organize_root_prompt(
        self, prompt_sample: BasicSamplePrompt, experiences: List[Tuple[str, str]]
    ):
        """Organizing the root prompt by adding experience."""
        # Create a new prompt sample to avoid changing the original one
        experienced_prompt_sample = BasicSamplePrompt(**prompt_sample)

        # Create a demo prompt derived from the prompt sample
        sample_demonstrations = BasicPromptFormat(
            **experienced_prompt_sample.demonstrations,
        )
        # Get the prompt of experiences
        if experiences:
            experience_prompt = self.organize_experience_prompt(
                experiences, with_flag=True
            )
            sample_demonstrations.tail = (
                f"""{sample_demonstrations.tail}\n\n{experience_prompt}"""
            )
        experienced_prompt_sample.demonstrations = sample_demonstrations

        return experienced_prompt_sample
