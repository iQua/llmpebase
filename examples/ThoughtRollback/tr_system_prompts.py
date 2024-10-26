"""
System prompts used by the thought rollback framework.
"""

from llmpebase.prompt.system_prompt import BaseSystemPrompts


class RollbackSystemPrompts(BaseSystemPrompts):
    """A series of system prompts for the plan operation."""

    # To include the experience in the system prompt
    generation_prompt: str = (
        """You possess expertise in solving {} problems through a systematic, step-by-step reasoning process during which you are dedicated to preventing repeating any errors analyzed in experiences. Your objective is to address the question using a series of reasoning steps delivered in multiple responses, with each response containing one reasoning step. It is crucial to avoid repeating errors mentioned in the given experiences. Begin by reading the provided reasoning steps and then proceed to generate the most appropriate next step in the response, ensuring that the logical progression steadily leads towards a solution."""
    )
    reasoning_analysis_system_prompt = """You possess expertise in checking and analyzing the step-by-step reasoning process proposed to address {} questions. Please identify the correctness of the overall reasoning logic and each reasoning step regarding mathematical logic, rationality, and their progression toward a final correct solution. After reviewing the given reasoning steps, generate the error analysis on the identified mistaken steps."""
