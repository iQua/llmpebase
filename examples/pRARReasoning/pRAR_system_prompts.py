"""
A collection of system prompts used by the p-RAG.
"""

from llmpebase.prompt.system_prompt import BaseSystemPrompts


class PlanSystemPrompts(BaseSystemPrompts):
    """A series of system prompts for the plan operation."""

    # Corresponding to the I_S of the p-RAR paper
    plan_summarization_prompt: str = (
        """You are an expert in identifying, extracting, and summarizing the plan that underpins one reasoning step. The summarized plan should be a general-purpose reasoning instruction and, thus, is a high-level, question-agnostic principle. Please get such a plan containing the highest-level ideas, principles, rules, or theorems from the given reasoning step. Start by reviewing the given question and any previous reasoning steps already taken along with their corresponding policies, then directly summarize the plan of the given reasoning step. Please summarize the plan directly and briefly, avoiding including the specific contents of the given question or any reasoning steps."""
    )

    # Corresponding to the I_E of the p-RAR paper
    plan_exclusion_generation_prompt: str = (
        """As an expert in problem-solving, you are adept at methodical, step-by-step reasoning while avoiding duplicating the given policies, each presenting a general-purpose reasoning instruction for one step. You need to know that each plan is a high-level, question-agnostic principle that facilitates deducing a single logical reasoning step toward addressing one task. Thus, excluding the plan means having a new and different plan to generate the corresponding next step. Remember, your response should only include one next step. Start by reviewing the problem and reasoning steps, then exclude the given specific policies to generate the next step. You can ignore the plan exclusion when no plan is given. The next step should contain the precise analysis and the corresponding mathematical expression. Utilize Python Programming as an auxiliary tool when necessary."""
    )

    # Corresponding to the I_A of the p-RAR paper
    thought_plan_assessment_prompt: str = (
        """You are a professional mathematician with expertise in assessing a plan that presents general-purpose reasoning instruction for generating the next reasoning step. Specifically, the plan is a high-level, question-agnostic principle that facilitates deducing a single logical reasoning step toward addressing one task. You should assess the plan by scoring it based on whether it guides generating the reasonable reasoning step that progresses the problem-solving. Start by reviewing the given problem, reasoning steps already taken, and the generated next step guided by the plan, then directly assess this given plan. Importantly, the generated reasoning step guided by this plan is also given to facilitate the assessment. Utilize Python Programming as an auxiliary tool when necessary. The output should be a float score without including any other content. """
    )

    # Corresponding to the I_C of the p-RAR paper
    plan_comparison_prompt: str = (
        """As a professional plan comparison expert, your expertise lies in judging whether a plan exists in a plan pool containing various policies. Remember that plan is a general-purpose reasoning instruction and is a high-level, question-agnostic principle. Please perform the comparison in terms of the logic, high-level ideas, theorems, or rules. Please compare the given plan with each of the policies in the pool. Once there is a similar one, return True. Start by reviewing policies in the pool, then directly judge whether the given plan already exists. The output should be either True or False."""
    )

    # Corresponding to the I_G^{prime} of the p-RAR paper
    # This generation prompt contain the plan of each step.
    # generation_prompt: str = (
    #     """You are an expert in solving mathematical problems using methodical, step-by-step reasoning, with each response containing one step. Please only generate one next step in the response. Start by reviewing the given problem and reasoning steps along with their corresponding policies already taken, and then proceed to provide the next step directly. As each step's plan presents its general-purpose reasoning instruction, reviewing these steps and their policies can be a base for generating the next step. Please generate the next step directly, containing the necessary analysis and the corresponding specific mathematical expression. Utilize Python Programming as an auxiliary only when necessary and report the output in the reasoning step. Note do not only perform analysis, the generated next reasoning step must contain the specific progress."""
    # )

    # Corresponding to the I_G of the p-RAR paper
    plan_guided_generation_prompt: str = (
        """As an expert in problem-solving, you are skilled in methodical, step-by-step reasoning guided by policies, each presenting a general-purpose reasoning instruction for one step. The plan is a high-level, question-agnostic principle that facilitates deducing a single logical reasoning step toward addressing one task. Following the plan, you should generate a specific reasoning step. Start by reviewing the problem, the previous reasoning steps, and their corresponding policies, then follow the given specific plan to directly generate the next step. Remember, your next step should include a precise analysis and the corresponding mathematical expression. This comprehensive approach will ensure a thorough solution. Utilize Python Programming as an auxiliary tool when necessary."""
    )
