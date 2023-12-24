"""
Generic components for prompting.
"""

from dataclasses import dataclass

from transformers.utils import ModelOutput as FieldFrozenContainer


@dataclass
class BasicPromptFormat(FieldFrozenContainer):
    """
    A basic structure for the prompt.
    """

    content: str
    head: str = None
    notice: str = None
    tail: str = None

    prompt: str = None

    def __str__(self):
        # Build the prompt by combing each part
        self.prompt = f"""{self.head}{self.content}{self.notice}{self.tail}"""
        return self.prompt


@dataclass
class BasicAnswerPromptFormat(BasicPromptFormat):
    """
    A basic structure for the prompt of answer.
    """

    groundtruth: str = None

    def __str__(self):
        # Build the prompt by combing each part
        self.prompt = (
            f"""{self.head}{self.notice}{self.content} {self.groundtruth}{self.tail}"""
        )
        return self.prompt


@dataclass
class BasicSamplePrompt(FieldFrozenContainer):
    """
    A basic structure for the prompt sample, which is used as the input
    for the Llm to perform the reasoning.
    """

    head: str = "Answer the question about the problem {}."
    notice: str = None
    solution_flag: str = None
    demonstrations: BasicPromptFormat = None
    question: BasicPromptFormat = None
    answer: BasicPromptFormat = None

    prompt: str = None

    def __str__(self):
        # Build the prompt by combing each part
        self.prompt = f"""{self.head} {self.notice}{self.demonstrations}\n{self.question}{self.answer}"""
        return self.prompt


@dataclass
class BasicThoughtPromptFormat(BasicPromptFormat):
    """
    A basic structure for the prompt of the thought generation.
    """

    target: str = None

    def __str__(self):
        # Build the prompt by combing each part

        self.prompt = (
            f"""{self.head}{self.content}{self.target}{self.notice}{self.tail}"""
        )
        return self.prompt
