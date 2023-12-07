"""
Implementation of Chain Of Thought [1].

[1]. Wei, et.al., Chain-of-Thought Prompting Elicits Reasoning in Large Language Models, 23.
"""

import reasoner

from llmpebase.pipeline import Pipeline


def _main():
    """The core function for model running."""

    pipeline = Pipeline(reasoner=reasoner.CoTReasoner)
    pipeline.setup()
    pipeline.load_data()
    pipeline.execute()


if __name__ == "__main__":
    _main()
