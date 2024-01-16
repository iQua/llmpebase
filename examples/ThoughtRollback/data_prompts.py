"""
The prompts designed for different datasets.
"""
from llmpebase.model.prompting.prompt_generic import (
    BasicThoughtPromptFormat,
)

rollback_controller_prompt_format = BasicThoughtPromptFormat(
    head="{}Toward addressing the given question, below is a reasoning process containing {} steps and the corresponding analysis: \n",
    content="\n{}\n{}\n{}\n",
    target="""Please summarize the analysis within {}, and thus within the given {} steps, extract the indexes of those unnecessary, wrong, illogical, unreasonable, or vague reasoning steps. Only output the indexes after '{}'. """,
    notice="Output None when no steps are given.\n",
    tail="",
    prompt="",
)


mathematical_reasoning_analysis_prompt_format = BasicThoughtPromptFormat(
    head="{}Toward addressing the given question, below is a reasoning process containing {} steps: \n",
    content="\n{}\n\n",
    target="""Double-check the reasoning process within {}, please analyze its overall and each step's correctness by checking whether they are mathematical logic and rationality. Please report an error when any step does not contain a clear mathematical expression. """,
    notice="Output empty string when no steps are given.\n",
    tail="",
    prompt="",
)

mathematical_solution_analysis_prompt_format = BasicThoughtPromptFormat(
    head="{}Toward addressing the given question, below is a reasoning process containing {} steps: \n",
    content="\n{}\n\n",
    target="""Double-check the reasoning process within {}, please analyze its overall and each step's correctness by checking whether they are mathematical logic, rationality; Please check whether the final solution can be used to answer the given question '{}'. Please specifically report an error when the variables in the final solution are not substituted with the given value. """,
    notice="Output empty string when no steps are given.\n",
    tail="",
    prompt="",
)

rollback_prompt_formats = {
    "Intermediate": mathematical_reasoning_analysis_prompt_format,
    "Sink": mathematical_solution_analysis_prompt_format,
}
