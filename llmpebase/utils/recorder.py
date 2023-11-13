"""
Implementation of recorders for saving results to files.
"""

import os
import logging
from typing import Dict, List, Union
from typing import OrderedDict as OrderedDictType
from collections import OrderedDict
import json


class DualExtensionRecoder:
    """
    A base class to record the samples and results.

    The core key of `record_pool` and `sample_pool` is the sample id, which
    is used to distinguish samples.
    Args:
        sample_pool: A `OrderedDict`, key of this dict is the sample id while
         the value is a list in which each item contains a variant sample,
         such as the sample but with various properties.
         In Q&A task, the list contains different answers to one question
         In VQA task, the list contains different captions to one image.
        record_pool: A `OrderedDict`, key of this dict is the sample id while
         the value is a list in which each item is a record, such as one
         obtained result of this sample.
         In Q&A task, each record contains the responses and results of the corresponding
          sample. Each record corresponds one sample in that under same `sample_id` of
          sample_pool.
    """

    def __init__(
        self,
        records_filename: str = "records",
        samples_filename: str = "samples",
        record_path: str = None,
        record_name: str = None,
        # whether append the existing records instead of overwriting them
        is_append: bool = False,
    ) -> None:
        self.records_filename = records_filename
        self.samples_filename = samples_filename

        self.record_path = os.getcwd() if record_path is None else record_path
        self.record_name = "records" if record_name is None else record_name

        self.record_dir_path = os.path.join(self.record_path, record_name)

        os.makedirs(self.record_dir_path, exist_ok=True)

        self.record_save_path = os.path.join(
            self.record_dir_path, self.records_filename + ".json"
        )
        self.sample_save_path = os.path.join(
            self.record_dir_path, self.samples_filename + ".json"
        )

        # a pool recording all results
        self.record_pool: OrderedDictType[str, list] = OrderedDict()
        # a pool recording all samples
        self.sample_pool: OrderedDictType[str, list] = OrderedDict()
        # Note, the length of these two pools should be the same
        # as they are corresponding to each other

        # the core item used to judge whether two samples/records
        # are the same
        # core_sample_item: the key of the item in the sample to verify
        #   whether two samples are the same
        # core_record_item: the key of the item in the record to verify
        #   whether two records are the same
        self.core_sample_item = None
        self.core_record_item = None
        self.set_check_items(None, None)

        if is_append:
            self.load_records()
            self.load_samples()

    def set_check_items(self, sample_check_item, record_check_item):
        """The default key to be checked for similarity measurement for
        record_pool and sample_pool."""
        self.core_sample_item = (
            "sample_content" if sample_check_item is None else sample_check_item
        )
        self.core_record_item = (
            "record_content" if record_check_item is None else record_check_item
        )

    def load_records(self):
        """Loading the records from the disk."""
        if os.path.exists(self.record_save_path):
            logging.info("%s exists.", self.record_save_path)
            logging.info("Loading the records from %s", self.record_save_path)
            with open(self.record_save_path, "r", encoding="utf-8") as file:
                self.record_pool = json.load(file)

        else:
            logging.info(
                "%s does not exist. Will create new one", self.sample_save_path
            )

    def load_samples(self):
        """Loading the records from the disk."""
        if os.path.exists(self.sample_save_path):
            logging.info("%s exists.", self.sample_save_path)
            logging.info("Loading the records from %s", self.sample_save_path)
            with open(self.sample_save_path, "r", encoding="utf-8") as file:
                self.sample_pool = json.load(file)

        else:
            logging.info(
                "%s does not exist. Will create new one", self.sample_save_path
            )

    def polish_string(self, input_str: str):
        """Polishing the input_str to reduce invalid white spaces."""
        return input_str.strip()

    def is_existed_sample_id(self, sample_id: str):
        """Judge whether the given sample_id exists in self.sample_pool."""
        if sample_id in self.sample_pool:
            return True

        return False

    def get_sample_idx(self, sample_id: str, dst_sample: str):
        """Get the index of `dst_sample` under the `sample_id` of sample_pool."""
        src_samples = self.sample_pool[sample_id]
        dst_value = self.polish_string(dst_sample[self.core_sample_item])
        is_sample_exists = [
            dst_value == self.polish_string(sample[self.core_sample_item])
            for sample in src_samples
        ]
        if any(is_sample_exists):
            return is_sample_exists.index(True)

        return -1

    def is_record_consistent(self, src_record: dict, dst_record: dict):
        """Judge whether the two records are the same."""
        dst_value = self.polish_string(dst_record[self.core_record_item])
        src_value = self.polish_string(src_record[self.core_record_item])
        if src_value == dst_value:
            return True

        return False

    def merge_record_to_pool(
        self, sample_id: str, group_sample_idx: int, new_record: dict
    ):
        """Merging one record to `group_sample_idx`-th record in the pool."""
        for record_item in self.record_pool[sample_id][group_sample_idx]:
            if isinstance(
                self.record_pool[sample_id][group_sample_idx][record_item],
                list,
            ):
                self.record_pool[sample_id][group_sample_idx][record_item].extend(
                    new_record[record_item]
                )

    def add_one_record(
        self,
        sample: dict,
        record: Dict[str, Union[str, List[Union[str, bool]]]],
        sample_id: str = "",
    ):
        """Adding one record.

        :param sample: A `dict` in which each key is one item name of the sample
        :param record: A `dict` in which each key is one item name of the record
        """
        sample_id = sample.pop(sample_id, None)
        sample_id = self.polish_string(sample_id)

        if self.is_existed_sample_id(sample_id=sample_id):
            group_sample_idx = self.get_sample_idx(sample_id, dst_sample=sample)
            # current sample exists
            if group_sample_idx != -1:
                # get the corresponding record
                corr_record = self.record_pool[sample_id][group_sample_idx]
                if not self.is_record_consistent(corr_record, record):
                    # append the new record
                    self.record_pool[sample_id].append(record)
                else:
                    self.merge_record_to_pool(sample_id, group_sample_idx, record)
            else:
                self.sample_pool[sample_id].append(sample)
                self.record_pool[sample_id].append(record)

        else:
            # add the new sample and record
            self.sample_pool[sample_id] = [sample]
            self.record_pool[sample_id] = [record]

    def save_records(self):
        """Saving the records to the disk."""

        with open(self.record_save_path, "w", encoding="utf-8") as file:
            json.dump(self.record_pool, file, indent=2)

        with open(self.sample_save_path, "w", encoding="utf-8") as file:
            json.dump(self.sample_pool, file, indent=2)
