"""
A running pipeline for the plan optimization phase of p-RAG.
"""

import random
import logging

from llmpebase.pipeline import Pipeline


class PlanOptimizationPipeline(Pipeline):
    """
    A pipeline to accumulate the experiences for the SnowBall.
    """

    def __init__(self, reasoner, *args):
        super().__init__(reasoner, *args)

        self.optimization_config = self.model_config["optimization"]

    def execute(self):
        """Execute the pipeline to obtain the results."""
        n_epochs = self.optimization_config["epochs"]
        num_train_samples = (
            len(self.trainset)
            if "num_train_samples" not in self.optimization_config
            else self.optimization_config["num_train_samples"]
        )

        num_total_train_samples = len(self.trainset)
        trainset_idxs = random.sample(range(num_total_train_samples), num_train_samples)
        logging.info("Training on %d samples.", len(trainset_idxs))

        for epoch in range(1, n_epochs + 1):
            for idx, sample in enumerate(self.trainset):
                if idx not in trainset_idxs:
                    continue
                sample_info = sample["auxiliary"]["sample_info"]
                sample_id = sample_info["sample_id"]
                record_name = f"Epoch {epoch}-Idx {idx}-ID<{sample_id}>"
                if record_name in self.exist_records:
                    continue
                # Forward the reasoning
                prompt_sample, groundtruth = self.data_prompter.create_prompt_sample(
                    sample=sample,
                    dataset=self.trainset,
                    config=self.model_config,
                )

                contents = self.reasoner.forward(
                    prompt_sample=prompt_sample,
                    sample_name=record_name,
                    sample_info=sample_info,
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

                # After getting the results, backpropagate the results
                # to update the plan tree
                self.reasoner.structure.backpropagation(measurements["num_correct"])

                # Save the plan tree to the plan base
                self.reasoner.plan_operator.save_plan_tree(
                    self.reasoner.structure.plan_tree, sample_info
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

                # Reset the reasoning after processing current sample
                self.reasoner.reset_reasoning()
