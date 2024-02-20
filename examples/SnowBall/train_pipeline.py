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
        batch_size = self.train_config["batch_size"]
        train_dataloader = DataLoader(
            self.trainset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: batch,
        )
        # Iterative get samples for training
        # One batch of samples is a list in which each item
        # is a tuple of (sample, (prompt_sample, groundtruth))
        # Perform the reasoning for training of SnowBall
        global_idx = 0
        for batch_samples in train_dataloader:
            batch_comparisons = []
            for sample in batch_samples:
                prompt_sample, groundtruth = self.data_prompter.create_prompt_sample(
                    sample, None, None
                )

                sample_id = sample["auxiliary"]["sample_info"]["sample_id"]

                record_name = f"{global_idx}-ID<{sample_id}>"
                contents = self.reasoner.forward(
                    prompt_sample=prompt_sample,
                    sample_name=record_name,
                )
                results = [
                    self.extractor.forward(
                        content,
                        solution_flag=self.data_prompter.solution_flag,
                        problem_name=sample["auxiliary"]["sample_info"][
                            "sample_problem"
                        ],
                        question=sample["question"],
                    )
                    for content in contents
                ]
                groundtruths = [groundtruth] * len(results)
                measurements = self.evaluator.forward(results, groundtruths)
                comparisons = measurements["matches"]

                batch_comparisons.append(comparisons)

                self.memorizer.accumulate_experiences(
                    sample, defined_reasoner=self.reasoner, comparisons=comparisons
                )

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

                global_idx += 1
                # Reset the reasoning after processing current sample
                self.reasoner.reset_reasoning()
