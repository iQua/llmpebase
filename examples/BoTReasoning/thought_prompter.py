"""
The thought prompter for the Boosting of Thoughts (BoT).
"""
from typing import List


from llmpebase.model.prompting.base import BasicSamplePrompt, BasicPromptFormat
from llmpebase.model.prompting.thought_prompt import ThoughtStructurePrompt


class BoTThoughtPrompter(ThoughtStructurePrompt):
    """
    A thought prompter to organize the thought prompts for the Boosting of Thoughts (BoT).
    """

    experience_flag: str = """################################################"""

    generation_system_prompt: str = """You are an expert at solving mathematical problems by performing step-by-step reasoning toward problem solving. Before solving the question, within '#########', you are given trial-and-error reasoning experience containing error reports and detailed advice on revising historical reasoning steps. Before answering, learn from experience to avoid making the same mistakes, follow correct reasoning, and polish the reasoning steps to generate better ones to solve the task."""

    experience_prompt = BasicPromptFormat(
        head="Recall historical reasoning experience (Ignore when experience is empty):\n",
        content="######### The {}-th Reasoning Experience with Comments #########\n{}\n",
        notice="",
        tail="Before generating reasoning steps to answer the question, consider the analysis and advice in the above experience to avoid making similar mistakes and produce better steps.\n\n",
        prompt="",
    )

    def organize_experience_prompt(
        self, experience_container: List[str], with_flag: bool = True
    ):
        """Organize experiences in the container into a single string."""
        experience_prompt = BasicPromptFormat(**self.experience_prompt)

        experience_chains = []
        for i, experience in enumerate(experience_container, start=1):
            experience_chains.append(
                self.experience_prompt.content.format(i, experience)
            )
        experience_prompt.content = "\n\n".join(experience_chains)
        if with_flag:
            experience_prompt.content = f"""{self.experience_flag}\n{experience_prompt.content}\n{self.experience_flag}\n\n"""

        return experience_prompt

    def organize_root_prompt(
        self, prompt_sample: BasicSamplePrompt, experiences: List[str]
    ):
        """Organizing the root prompt by adding experience."""
        # Get the prompt of experiences
        experience_prompt = self.organize_experience_prompt(experiences, with_flag=True)

        # Create a answer prompt derived from the prompt sample
        sample_demonstrations = BasicPromptFormat(
            **prompt_sample.demonstrations,
        )
        sample_demonstrations.tail = (
            f"""{sample_demonstrations.tail}\n\n{experience_prompt}"""
        )
        prompt_sample.demonstrations = sample_demonstrations

        return prompt_sample
