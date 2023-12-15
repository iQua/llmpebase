"""
An implementation of Boosting of Thoughts (BoT).
"""


import BoT_reasoner

from llmpebase.pipeline import Pipeline


def _main():
    """The core function for model running."""
    bot = BoT_reasoner.BoostOfThoughts()

    pipeline = Pipeline(reasoner=bot)
    pipeline.setup()
    pipeline.load_data()
    pipeline.execute()


if __name__ == "__main__":
    _main()
