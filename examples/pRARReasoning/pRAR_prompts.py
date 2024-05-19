"""
Policy and Thought Prompts for the p-RAR approach.
"""

from llmpebase.prompt.thought_prompt import BaseThoughtPrompts, ThoughtGenerationPrompts
from llmpebase.prompt.generic import BasicThoughtPromptFormat


class PolicyPrompts:
    """A series of prompts for the policy generation and creation."""

    # # Corresponding to the I_S of the p-RAR paper
    # HOLD FOR POSSIBLE FUTURE USE
    # policy_summarization_start_flag: str = """<Policy Summarization>"""
    # policy_summarization_end_flag: str = """<\\Policy Summarization>"""

    # Corresponding to the I_E of the p-RAR paper
    policy_exclusion_start_flag: str = """<Exclusion Policies>"""
    policy_exclusion_end_flag: str = """<\\Exclusion Policies>"""

    # Corresponding to the I_A of the p-RAR paper
    policy_assessment_start_flag: str = """<Policy Assessment>"""
    policy_assessment_end_flag: str = """<\\Policy Assessment>"""

    # Corresponding to the I_C of the p-RAR paper
    policy_comparison_start_flag: str = """<Policy Pool>"""
    policy_comparison_end_flag: str = """<\\Policy Pool>"""

    # Corresponding to the I_G^{prime} and I_G of the p-RAR paper
    policy_chain_start_flag: str = """<Policy Chain>"""
    policy_chain_end_flag: str = """<\\Policy Chain>"""

    policy_start_flag: str = """<Policy>"""
    policy_end_flag: str = """<\\Policy>"""

    policy_guided_thought_chain_start_flag: str = """<Step>"""
    policy_guided_thought_chain_end_flag: str = """<\\Step>"""

    # The head of each step
    policy_head: str = "Policy {}."

    # Corresponding to the I_S of the p-RAR paper
    # Braces: 1). Question, 2) First Reasoning Step
    first_policy_summarization_prompt = BasicThoughtPromptFormat(
        head="\n\n{}\nLet's focus on summarize the policy that underpins the first reasoning step.\n",
        content="\n{}\n",
        target="Please review Step 1 within {} and summarize its policy, i.e., Policy 1.",
        notice=" Only direct output summarized policy. Do not include the Policy index in the output.",
        tail="",
        prompt="",
    )
    # Corresponding to the I_S of the p-RAR paper
    # Braces: 1). Question, 2) Step index,
    # 3) Reasoning Chain, 4) Policy Chain, 5) Policy thought
    # 6) Chain flag, 7) Policy flag Step index, 8) Step idx, 9) Policy thought flag, 10). Policy index
    policy_summarization_prompt = BasicThoughtPromptFormat(
        head="\n\n{}\nLet's focus on summarize the policy that underpins the reasoning step {}.\n",
        content="\n{}\n{}\n\n{}",
        target="Please review the reasoning steps within {} and their corresponding policies within {} and proceed to summarize the policy of Step {} within {}, i.e., Policy {}.",
        notice=" Only direct output summarized policy. Do not include the Policy index in the output.",
        tail="",
        prompt="",
    )

    # Corresponding to the I_A of the p-RAR paper
    # Braces: 1) Question
    #   2) Generated Thought, 3) Policy Assessment
    #   3) Policy candidates flag
    assess_first_policy_prompt = BasicThoughtPromptFormat(
        head="{}Let's focus on assessing whether the policy guides the generation of an effective first step.\n",
        content="{}\n{}\n\n",
        target="Please assess Policy {} within {}. Notice that the reasoning Step 1 within {} is guided by the Policy 1 within {}.",
        notice=" Only output the assessment score ranging from 0 to 1, while a higher score means a better policy as reasoning guidance.",
        tail="",
        prompt="",
    )
    # Corresponding to the I_A of the p-RAR paper
    # Braces: 1) Question
    #   3) Reasoning chain 5) Generated Thought 6) Policy Assessment
    #   7) Reasoning chain flag
    assess_next_policy_prompt = BasicThoughtPromptFormat(
        head="{}Let's focus on assessing whether the policy can guide the generation of an effective next reasoning step.\n",
        content="\n{}\n\n{}\n{}",
        target="Please review the reasoning steps within the tag {} and the generated next Step {} within {} guided by the Policy {} within the tag {}, then assess this Policy {}.",
        notice=" Only output the assessment score ranging from 0 to 1, while a higher score means a better policy as reasoning guidance.",
        tail="",
        prompt="",
    )

    # Corresponding to the I_C of the p-RAR paper
    # Braces: 1) Question 2) Step index
    #   3) Policy chain 4) Policy pool
    #   5) Policy index, 6) Policy pool flag 7) Policy index
    compare_policy_prompt = BasicThoughtPromptFormat(
        head="Let's focus on whether the given policy exists in the policy pool.\n",
        content="\n{}\n\n{}",
        target="Please judge whether the Policy {} within the tag {} already exists in the policies within the tag {}.",
        notice="Only output True if exists, or False if not.",
        tail="",
        prompt="",
    )


class PolicyThoughtGenerationPrompts(ThoughtGenerationPrompts):
    """
    A base class to organize the prompt with the policy for the thought generation.
    """

    # Corresponding to the I_G^{prime} of the p-RAR paper
    #  Braces: 1) Question, 2) Reasoning chain, 3) Policy chain
    #   4) Reasoning chain flag, 5) Policy chain flag 6) Step index
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
    # Braces: 1) Question, 2) Policy, 3) Policy flag
    policy_guide_first_step_prompt = BasicThoughtPromptFormat(
        head="{}Let's focus on following the policy to directly generating the first reasoning step.\n",
        content="\n{}\n\n",
        target="Please follow the Policy 1 provided within the tag {} to generate a well-crafted first step, i.e., Step 1.",
        notice=" Only output the generated step. Do not include the Step index in the output.",
        tail="",
        prompt="",
    )
    # Braces: 1). Question,
    # 2) Reasoning chain, 3) Policy chain 4) Policy
    # 5) Reasoning chain flag, 6) Policy chain flag, 6) Policy index,
    # 7) Policy flag 8) Step index
    policy_guide_next_step_prompt = BasicThoughtPromptFormat(
        head="{}Let's focus on following the policy to directly generate the next possible reasoning step for the reasoning steps below.\n",
        content="\n{}\n{}\n\n{}\n\n",
        target="Please review the reasoning steps within the tag {} along with their policies within the tag {}, then follow the Policy {} within the tag {} to proceed to directly generate the best next step, i.e., Step {}.",
        notice=" Only output the generated step. Do not include the Step index in the output.",
        tail="",
        prompt="",
    )

    # Corresponding to the I_E of the p-RAR paper
    # Braces: 1) Question
    #   2). Policy exclusion
    #   4) Policy exclusion flag
    exclusive_policy_first_step_prompt = BasicThoughtPromptFormat(
        head="{}Let's focus on avoiding repeating the given policies to directly generate the first reasoning step to start addressing the question.\n",
        content="\n{}\n\n",
        target="As the start of reasoning, please exclude Policy 1 listed within the tag {} to generate a well-crafted first step, i.e., Step 1.",
        notice=" Only output the generated step. Do not include the Step index in the output.",
        tail="",
        prompt="",
    )
    # Braces: 1) Question,
    #   2) Reasoning chain, 3) Policy chain 4) Policy Exclusion
    #   5) Reasoning chain flag, 6) Policy chain flag, 7) Policy candidate index
    #   8) Policy candidate flag 9) Step index
    exclusive_policy_next_step_prompt = BasicThoughtPromptFormat(
        head="{}Let's focus on avoiding using the given policies to carefully and directly generate the next possible reasoning step for the reasoning steps below.\n",
        content="\n{}\n{}\n\n{}",
        target="Please review the reasoning steps within the tag {} and their policies within the tag {}, then specifically avoid repeating all Policy {} listed within tag {} to proceed to directly generate the best next step, i.e., Step {}.",
        notice=" Only output the generated step. Do not include the Step index in the output.",
        tail="",
        prompt="",
    )


class BasePolicyThoughtPrompts(BaseThoughtPrompts):
    """A base class to organize the policy-based thought prompts"""

    generation: PolicyThoughtGenerationPrompts = PolicyThoughtGenerationPrompts()
