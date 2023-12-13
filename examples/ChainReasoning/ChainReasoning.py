"""
A reasoning process organized as a chain thought structure.
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
