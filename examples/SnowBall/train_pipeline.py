"""
The train pipeline of the SnowBall.

"""

from torch.utils.data import DataLoader

from llmpebase.pipeline import Pipeline

from memorizer import LLMMemorizer

from llmpebase.config import Config


class ExperienceAccumulationPipeline(Pipeline):
    """
    A pipeline to accumulate the experiences for the SnowBall.
    """

    def __init__(self, reasoner, memorizer: LLMMemorizer = None, *args):
        super().__init__(reasoner, *args)

        self.memorizer = (
            LLMMemorizer(model_config=self.model_config, log_config=self.log_config)
            if memorizer is None
            else memorizer
        )
        self.train_config = Config.items_to_dict(Config().train._asdict())

    def execute(self):
        """Execute the pipeline to obtain the results."""
        n_epochs = self.train_config["epochs"]
        for epoch in range(1, n_epochs + 1):
            for idx, sample in enumerate(self.trainset):
                sample_info = sample["auxiliary"]["sample_info"]
                sample_id = sample_info["sample_id"]
                record_name = f"{epoch}-{idx}-ID<{sample_id}>"
                if record_name in self.exist_records:
                    continue
                prompt_sample, groundtruth = self.data_prompter.create_prompt_sample(
                    sample=sample,
                    dataset=self.trainset,
                    config=self.model_config,
                )
                contents = self.reasoner.forward(
                    prompt_sample=prompt_sample,
                    sample_name=record_name,
                )
                results = [
                    self.extractor.forward(
                        content,
                        solution_flag=self.data_prompter.solution_flag,
                        problem_name=sample_info["sample_problem"],
                        question=sample["question"],
                    )
                    for content in contents
                ]
                groundtruths = [groundtruth] * len(results)
                measurements = self.evaluator.forward(results, groundtruths)
                comparisons = measurements["matches"]

                output = {
                    "request_prompt": str(prompt_sample),
                    "responses": contents,
                    "groundtruth": groundtruths,
                    "results": results,
                    "measurements": measurements,
                }

                self.recorder.save_one_record(
                    sample=sample,
                    output=output,
                    statistic=self.reasoner.get_cost_statistics(latest=True),
                    sample_name=record_name,
                )

                # Memory the experiences
                self.memorizer.memory(
                    sample=sample,
                    solution_chains=self.reasoner.solution_extractor.extract_solution_chains(
                        self.reasoner.structure
                    ),
                    comparisons=comparisons,
                )

                # Reset the reasoning after processing current sample
                self.reasoner.reset_reasoning()
