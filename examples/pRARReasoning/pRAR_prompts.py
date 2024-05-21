"""
Plan and Thought Prompts for the p-RAR approach.
"""

from llmpebase.prompt.thought_prompt import BaseThoughtPrompts, ThoughtGenerationPrompts
from llmpebase.prompt.generic import BasicThoughtPromptFormat


class PlanPrompts:
    """A series of prompts for the plan generation and creation."""

    # # Corresponding to the I_S of the p-RAR paper
    # HOLD FOR POSSIBLE FUTURE USE
    # plan_summarization_start_flag: str = """<Plan Summarization>"""
    # plan_summarization_end_flag: str = """<\\Plan Summarization>"""

    # Corresponding to the I_E of the p-RAR paper
    plan_exclusion_start_flag: str = """<Exclusion Policies>"""
    plan_exclusion_end_flag: str = """<\\Exclusion Policies>"""

    # Corresponding to the I_A of the p-RAR paper
    plan_assessment_start_flag: str = """<Plan Assessment>"""
    plan_assessment_end_flag: str = """<\\Plan Assessment>"""

    # Corresponding to the I_C of the p-RAR paper
    plan_comparison_start_flag: str = """<Plan Pool>"""
    plan_comparison_end_flag: str = """<\\Plan Pool>"""

    # Corresponding to the I_G^{prime} and I_G of the p-RAR paper
    plan_chain_start_flag: str = """<Plan Chain>"""
    plan_chain_end_flag: str = """<\\Plan Chain>"""

    plan_start_flag: str = """<Plan>"""
    plan_end_flag: str = """<\\Plan>"""

    # The head of each step
    plan_head: str = "Plan {}."

    step_start_flag: str = """<Step>"""
    step_end_flag: str = """<\\Step>"""

    # Corresponding to the I_S of the p-RAR paper
    # Braces: 1). Question, 2) First Reasoning Step
    first_plan_summarization_prompt = BasicThoughtPromptFormat(
        head="{}\nFor the given question, let's focus on summarize the plan that underpins the first reasoning step.\n",
        content="\n{}\n\n",
        target="Please review Step 1 within {} and summarize its plan, i.e., Plan 1.",
        notice=" Only direct output summarized plan. Do not include the Plan index in the output. Remember that the plan is a high-level, question-agnostic principle. Do not include any question or reasoning step content in the plan.",
        tail="",
        prompt="",
    )
    # Corresponding to the I_S of the p-RAR paper
    # Braces: 1). Question, 2) Step index,
    # 3) Reasoning Chain, 4) Plan Chain, 5) Plan thought
    # 6) Chain flag, 7) Plan flag Step index, 8) Step idx, 9) Plan thought flag, 10). Plan index
    plan_summarization_prompt = BasicThoughtPromptFormat(
        head="{}\nFor the given question, let's focus on summarize the plan that underpins the reasoning step {}.\n",
        content="\n{}\n{}\n\n{}\n\n",
        target="Please review the reasoning steps within {} and their corresponding policies within {} and proceed to summarize the plan of Step {} within {}, i.e., Plan {}.",
        notice=" Only direct output summarized plan. Do not include the Plan index in the output. Remember that the plan is a high-level, question-agnostic principle. Do not include any question or reasoning step content in the plan.",
        tail="",
        prompt="",
    )

    # Corresponding to the I_A of the p-RAR paper
    # Braces: 1) Question
    #   2) Generated Thought, 3) Plan Assessment
    #   3) Plan candidates flag
    assess_first_plan_prompt = BasicThoughtPromptFormat(
        head="{}\nFor the given question, let's focus on assessing whether the plan guides the generation of an effective first step.\n",
        content="{}\n{}\n\n",
        target="Please assess Plan {} within {}. Notice that the reasoning Step 1 within {} is guided by the Plan 1 within {}.",
        notice=" Only output the assessment score ranging from 0 to 1, while a higher score means a better plan as reasoning guidance.",
        tail="",
        prompt="",
    )
    # Corresponding to the I_A of the p-RAR paper
    # Braces: 1) Question
    #   3) Reasoning chain 5) Generated Thought 6) Plan Assessment
    #   7) Reasoning chain flag
    assess_next_plan_prompt = BasicThoughtPromptFormat(
        head="{}\nFor the given question, Let's focus on assessing whether the plan can guide the generation of an effective next reasoning step.\n",
        content="\n{}\n\n{}\n{}\n\n",
        target="Please review the reasoning steps already taken within the tag {} and the generated next Step {} within {} guided by the Plan {} within the tag {}, then assess this Plan {}.",
        notice=" Only output the assessment score ranging from 0 to 1, while a higher score means a better plan as reasoning guidance.",
        tail="",
        prompt="",
    )

    # Corresponding to the I_C of the p-RAR paper
    # Braces: 1) Question 2) Step index
    #   3) Plan chain 4) Plan pool
    #   5) Plan index, 6) Plan pool flag 7) Plan index
    compare_plan_prompt = BasicThoughtPromptFormat(
        head="Let's focus on whether the given plan exists in the plan pool.\n",
        content="\n{}\n\n{}\n\n",
        target="Please judge whether the Plan {} within the tag {} already exists in the policies within the tag {}.",
        notice="Only output True if exists, or False if not. Remember that plan is a high-level, question-agnostic principle. Do not focus on text details but on the logic, high-level ideas, theorems, or rules.",
        tail="",
        prompt="",
    )


class PlanThoughtGenerationPrompts(ThoughtGenerationPrompts):
    """
    A base class to organize the prompt with the plan for the thought generation.
    """

    # Corresponding to the I_G^{prime} of the p-RAR paper
    #  Braces: 1) Question, 2) Reasoning chain, 3) Plan chain
    #   4) Reasoning chain flag, 5) Plan chain flag 6) Step index
    # next_step_prompt = BasicThoughtPromptFormat(
    #     head="{}Let's focus on carefully and directly generating the next possible reasoning step for the reasoning steps below.\n",
    #     content="\n{}\n\n{}\n\n",
    #     target="Please review the reasoning steps within the tag {} along with their policies within the tag {}, then proceed to directly generate the best next step, i.e., Step {}.",
    #     notice=" Only output the generated step. Do not include the Step index in the output.",
    #     tail="",
    #     prompt="",
    # )

    next_step_prompt = BasicThoughtPromptFormat(
        head="{}Let's focus on carefully and directly generating the next possible reasoning step for the reasoning steps below.\n",
        content="\n{}\n\n",
        target="Please review the reasoning steps within the tag {}, then proceed to directly generate the best next step, i.e., Step {}.",
        notice=" Only output the generated step. Do not include the Step index in the output.",
        tail="",
        prompt="",
    )

    # Corresponding to the I_G of the p-RAR paper
    # Braces: 1) Question, 2) Plan, 3) Plan flag
    plan_guide_first_step_prompt = BasicThoughtPromptFormat(
        head="{}Let's focus on following the plan to directly generating the first reasoning step.\n",
        content="\n{}\n\n",
        target="Please follow the Plan 1 provided within the tag {} to generate a well-crafted first step, i.e., Step 1.",
        notice=" Only output the generated step. Do not include the Step index in the output.",
        tail="",
        prompt="",
    )
    # Braces: 1). Question,
    # 2) Reasoning chain, 3) Plan chain 4) Plan
    # 5) Reasoning chain flag, 6) Plan chain flag, 6) Plan index,
    # 7) Plan flag 8) Step index
    plan_guide_next_step_prompt = BasicThoughtPromptFormat(
        head="{}Let's focus on following the plan to directly generate the next reasoning step for the reasoning steps below.\n",
        content="\n{}\n{}\n\n{}\n\n",
        target="Please review the reasoning steps within the tag {} along with their policies within the tag {}, then follow the Plan {} within the tag {} to proceed to directly generate the best next step, i.e., Step {}.",
        notice=" Only output the generated step. Do not include the Step index in the output.",
        tail="",
        prompt="",
    )

    # Corresponding to the I_E of the p-RAR paper
    # Braces: 1) Question
    #   2). Plan exclusion
    #   4) Plan exclusion flag
    exclusive_plan_first_step_prompt = BasicThoughtPromptFormat(
        head="{}Let's focus on avoiding repeating the given policies to directly generate the first reasoning step to start addressing the question.\n",
        content="\n{}\n\n",
        target="As the start of reasoning, please exclude Plan 1 listed within the tag {} to generate a well-crafted first step, i.e., Step 1.",
        notice=" Only output the generated step. Do not include the Step index in the output.",
        tail="",
        prompt="",
    )
    # Braces: 1) Question,
    #   2) Reasoning chain, 3) Plan chain 4) Plan Exclusion
    #   5) Reasoning chain flag, 6) Plan chain flag, 7) Plan candidate index
    #   8) Plan candidate flag 9) Step index
    exclusive_plan_next_step_prompt = BasicThoughtPromptFormat(
        head="{}Let's focus on avoiding using the given policies to carefully and directly generate the next possible reasoning step for the reasoning steps below.\n",
        content="\n{}\n{}\n\n{}\n\n",
        target="Please review the reasoning steps within the tag {} and their policies within the tag {}, then specifically avoid repeating all Plan {} listed within tag {} to proceed to directly generate the best next step, i.e., Step {}.",
        notice=" Only output the generated step. Do not include the Step index in the output.",
        tail="",
        prompt="",
    )


class BasePlanThoughtPrompts(BaseThoughtPrompts):
    """A base class to organize the plan-based thought prompts"""

    generation: PlanThoughtGenerationPrompts = PlanThoughtGenerationPrompts()
