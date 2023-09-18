"""
Reading runtime parameters from a standard configuration file (which is easier
to work on than JSON).
"""
import sys
import time
import argparse
import json
import logging
import os
from pathlib import Path
from collections import OrderedDict, namedtuple
from typing import Any, IO
import shutil

import torch
import numpy as np
import yaml


class Loader(yaml.SafeLoader):
    """YAML Loader with `!include` constructor."""

    def __init__(self, stream: IO) -> None:
        """Initialise Loader."""

        try:
            self.root_path = os.path.split(stream.name)[0]
        except AttributeError:
            self.root_path = os.path.curdir

        super().__init__(stream)


class Config:
    """
    Retrieving configuration parameters by parsing a configuration file
    using the YAML configuration file parser.
    """

    _instance = None

    @staticmethod
    def construct_include(loader: Loader, node: yaml.Node) -> Any:
        """Include file referenced at node."""

        filename = os.path.abspath(
            os.path.join(loader.root_path, loader.construct_scalar(node))
        )
        extension = os.path.splitext(filename)[1].lstrip(".")

        with open(filename, "r", encoding="utf-8") as config_file:
            if extension in ("yaml", "yml"):
                return yaml.load(config_file, Loader)
            elif extension in ("json",):
                return json.load(config_file)
            else:
                return "".join(config_file.readlines())

    @staticmethod
    def construct_join(loader: Loader, node: yaml.Node) -> Any:
        """Support os.path.join at node."""
        seq = loader.construct_sequence(node)
        return "/".join([str(i) for i in seq])

    def __new__(cls):
        if cls._instance is None:
            parser = argparse.ArgumentParser()
            parser.add_argument("-i", "--id", type=str, help="Unique client ID.")
            parser.add_argument(
                "-p", "--port", type=str, help="The port number for running a server."
            )
            parser.add_argument(
                "-c",
                "--config",
                type=str,
                default="./config.yml",
                help="Federated learning configuration file.",
            )
            parser.add_argument(
                "-b",
                "--base",
                type=str,
                default="./",
                help="The base path for datasets and models.",
            )
            parser.add_argument(
                "-s",
                "--server",
                type=str,
                default=None,
                help="The server hostname and port number.",
            )
            parser.add_argument(
                "-d",
                "--download",
                action="store_true",
                help="Download the dataset to prepare for a training session.",
            )
            parser.add_argument(
                "-r",
                "--resume",
                action="store_true",
                help="Resume a previously interrupted training session.",
            )
            parser.add_argument(
                "-l", "--log", type=str, default="info", help="Log messages level."
            )

            args = parser.parse_args()
            Config.args = args

            if Config.args.id is not None:
                Config.args.id = int(args.id)
            if Config.args.port is not None:
                Config.args.port = int(args.port)

            cls._instance = super(Config, cls).__new__(cls)

            if "config_file" in os.environ:
                filename = os.environ["config_file"]
            else:
                filename = args.config

            yaml.add_constructor("!include", Config.construct_include, Loader)
            yaml.add_constructor("!join", Config.construct_join, Loader)

            if os.path.isfile(filename):
                with open(filename, "r", encoding="utf-8") as config_file:
                    config = yaml.load(config_file, Loader)
            else:
                # if the configuration file does not exist, raise an error
                raise ValueError("A configuration file must be supplied.")

            Config.environment = Config.namedtuple_from_dict(config["environment"])
            Config.data = Config.namedtuple_from_dict(config["data"])
            Config.model = Config.namedtuple_from_dict(config["model"])
            Config.train = Config.namedtuple_from_dict(config["train"])
            Config.evaluation = Config.namedtuple_from_dict(config["evaluation"])
            Config.logging = Config.namedtuple_from_dict(config["logging"])

            HOMEPATH = str(Path.home())

            # Customizable dictionary of global parameters
            Config.params: dict = {}

            # A run ID is unique in an experiment
            # The precisions are unique for all parts of the model
            Config.params["run_id"] = os.getpid()

            # Project dir
            # The base path used for all datasets, models, checkpoints, and results
            Config.params["base_path"] = Config.args.base.replace("~", HOMEPATH)

            project_dir = Config.params["base_path"]

            # Get the name of the model
            Config.params["model_name"] = Config().model.model_name

            # Get the name of the data
            Config.params["data_name"] = Config().data.data_name

            # the basic saving dir name in
            # models/checkpoints/results/loggings
            basic_dir_name = Config.make_basic_save_path()

            # data dir
            # we hope the data can be placed under the project dir directly
            #   instead of be placing under project_dir/project_name
            data_path = Config().data.data_path.replace("~", HOMEPATH)
            if not os.path.exists(data_path):
                Config.data = Config.data._replace(
                    data_path=Path(os.path.join(project_dir, "data"))
                )

            if hasattr(Config().data, "datasource_path") and (
                Config().data.datasource_path is not None
            ):
                # if the set dir is not correct, we redirect to
                #   the default dir {data_path}/datasource_path
                datasource_path = Config().data.datasource_path.replace("~", HOMEPATH)
                if not os.path.exists(datasource_path):
                    expected_datasource_path = os.path.join(
                        project_dir, datasource_path
                    )

                    Config.data = Config.data._replace(
                        datasource_path=Path(expected_datasource_path).as_posix()
                    )

            # now = datetime.datetime.now()
            # now_time_str = (
            #     now.strftime("%Y-%m-%d %H:%M:%S").replace(" ", "-").replace(":", "-")
            # )
            # Logging dir
            # The 'experiments_dir' should be set in the path defined by -b
            # as the project_dir will be utilized as the experiments_dir
            # directly
            experiments_dir = Config().logging.experiments_dir.replace("~", HOMEPATH)
            if not os.path.exists(experiments_dir):
                desired_path = Path(project_dir).as_posix()
                Config.logging = Config.logging._replace(experiments_dir=desired_path)

            # Resume checkpoint
            data_checkpoints_dir = os.path.join(
                Config().logging.checkpoints_dir, basic_dir_name
            ).replace("~", HOMEPATH)
            if not os.path.exists(data_checkpoints_dir):
                Config.logging = Config.logging._replace(
                    checkpoints_dir=Path(
                        os.path.join(
                            Config().logging.experiments_dir,
                            "checkpoints",
                            basic_dir_name,
                        )
                    ).as_posix()
                )
            else:
                Config.logging = Config.logging._replace(
                    checkpoints_dir=data_checkpoints_dir
                )

            data_results_dir = os.path.join(
                Config().logging.results_dir, basic_dir_name
            ).replace("~", HOMEPATH)
            if not os.path.exists(data_results_dir):
                Config.logging = Config.logging._replace(
                    results_dir=Path(
                        os.path.join(
                            Config().logging.experiments_dir, "results", basic_dir_name
                        )
                    ).as_posix()
                )
            else:
                Config.logging = Config.logging._replace(results_dir=data_results_dir)

            # Loggings
            loggings_dir = os.path.join(
                Config().logging.loggings_dir, basic_dir_name
            ).replace("~", HOMEPATH)
            if not os.path.exists(loggings_dir):
                Config.logging = Config.logging._replace(
                    loggings_dir=Path(
                        os.path.join(
                            Config().logging.experiments_dir, "loggings", basic_dir_name
                        )
                    ).as_posix()
                )
            else:
                Config.logging = Config.logging._replace(loggings_dir=loggings_dir)

            # Visual results
            data_visualizations_dir = os.path.join(
                Config().logging.visualizations_dir, basic_dir_name
            ).replace("~", HOMEPATH)
            if not os.path.exists(data_visualizations_dir):
                Config.logging = Config.logging._replace(
                    visualizations_dir=Path(
                        os.path.join(
                            Config().logging.experiments_dir,
                            "visualizations",
                            basic_dir_name,
                        )
                    ).as_posix()
                )
            else:
                Config.logging = Config.logging._replace(
                    visualizations_dir=data_visualizations_dir
                )

            # Pretrained models
            if not os.path.exists(Config().model.pretrained_models_dir):
                Config.model = Config.model._replace(
                    pretrained_models_dir=Path(
                        os.path.join(
                            Config().logging.experiments_dir, "pretrained_models"
                        )
                    ).as_posix()
                )

            os.makedirs(Config().logging.experiments_dir, exist_ok=True)
            os.makedirs(Config().logging.checkpoints_dir, exist_ok=True)
            os.makedirs(Config().logging.results_dir, exist_ok=True)
            os.makedirs(Config().logging.visualizations_dir, exist_ok=True)
            os.makedirs(Config().logging.loggings_dir, exist_ok=True)
            os.makedirs(Config().model.pretrained_models_dir, exist_ok=True)

            # Saving the given config file to the corresponding
            # results/models/checkpoints
            config_file_name = os.path.basename(filename)
            config_log_path = os.path.join(
                Config().logging.loggings_dir, config_file_name
            )
            config_checkpoint_path = os.path.join(
                Config().logging.checkpoints_dir, config_file_name
            )
            config_result_path = os.path.join(
                Config().logging.results_dir, config_file_name
            )
            for target_path in [
                config_log_path,
                config_checkpoint_path,
                config_result_path,
            ]:
                if not os.path.exists(target_path):
                    shutil.copyfile(src=filename, dst=target_path)

            # add the log file
            # thus, presenting logging info to the file

            if hasattr(Config().logging, "basic_log_type"):
                basic_log_type = Config().logging.basic_log_type
            else:
                basic_log_type = "info"

            numeric_level = getattr(logging, basic_log_type.upper(), None)

            if not isinstance(numeric_level, int):
                raise ValueError(f"Invalid log level: {basic_log_type}")

            formatter = logging.Formatter(
                fmt="[%(levelname)s][%(asctime)s]: %(message)s", datefmt="%H:%M:%S"
            )

            root_logger = logging.getLogger()
            root_logger.setLevel(numeric_level)

            # only print to the screen
            if hasattr(Config().logging, "stdout_log_type"):
                stdout_log_type = Config().logging.stdout_log_type
                stdout_log_numeric_level = getattr(
                    logging, stdout_log_type.upper(), None
                )
                stdout_handler = logging.StreamHandler(sys.stdout)
                stdout_handler.setLevel(stdout_log_numeric_level)
                stdout_handler.setFormatter(formatter)
                root_logger.addHandler(stdout_handler)

            if hasattr(Config().logging, "file_log_type"):
                file_log_type = Config().logging.file_log_type
                file_log_numeric_level = getattr(logging, file_log_type.upper(), None)
                log_name = time.strftime("%Y_%m_%d__%H_%M_%S.txt", time.localtime())
                log_file_name = os.path.join(
                    Config().logging.loggings_dir, file_log_type + "_" + log_name
                )

                file_handler = logging.FileHandler(log_file_name)
                file_handler.setLevel(file_log_numeric_level)
                file_handler.setFormatter(formatter)

                root_logger.addHandler(file_handler)

        return cls._instance

    @staticmethod
    def namedtuple_from_dict(obj):
        """Creates a named tuple from a dictionary."""
        if isinstance(obj, dict):
            fields = sorted(obj.keys())
            namedtuple_type = namedtuple(
                typename="Config", field_names=fields, rename=True
            )
            field_value_pairs = OrderedDict(
                (str(field), Config.namedtuple_from_dict(obj[field]))
                for field in fields
            )
            try:
                return namedtuple_type(**field_value_pairs)
            except TypeError:
                # Cannot create namedtuple instance so fallback to dict (invalid attribute names)
                return dict(**field_value_pairs)
        elif isinstance(obj, (list, set, tuple, frozenset)):
            return [Config.namedtuple_from_dict(item) for item in obj]
        else:
            return obj

    @staticmethod
    def device() -> str:
        """Returns the device to be used for training."""
        device = "cpu"

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            if hasattr(Config().environment, "device_id"):
                device = "cuda:" + str(Config().environment.device_id)
            else:  # randomly select one cuda for training
                device = "cuda:" + str(np.random.randint(0, torch.cuda.device_count()))

        return device

    @staticmethod
    def is_parallel() -> bool:
        """Check if the hardware and OS support data parallelism."""

        return (
            hasattr(Config().environment, "distributed")
            and Config().environment.parallelized
            and torch.cuda.is_available()
            and torch.distributed.is_available()
            and torch.cuda.device_count() > 1
        )

    @staticmethod
    def items_to_dict(base_dict) -> dict:
        """Convert items of the dict to be dict if possible.

        The main purpose of this function is to address the
        condition of nested Config term, such as
        {"key1": Config(k11=5, k12=7),
         "key2": Config(k12=5, k22=Config(m=1, n=7))
         }
        """
        for key, value in base_dict.items():
            if not isinstance(value, dict):
                if hasattr(value, "_asdict"):
                    value = value._asdict()
                    value = Config().items_to_dict(value)
                    base_dict[key] = value
            else:
                value = Config().items_to_dict(value)
                base_dict[key] = value
        return base_dict

    @staticmethod
    def to_dict() -> dict:
        """Converting the current run-time configuration to a dict."""

        config_data = dict()
        config_data["train"] = Config.train._asdict()
        config_data["environment"] = Config.environment._asdict()
        config_data["data"] = Config.data._asdict()
        config_data["model"] = Config.model._asdict()
        config_data["logging"] = Config.logging._asdict()
        config_data["evaluation"] = Config.evaluation._asdict()
        for term in ["train", "environment", "data", "model", "logging", "evaluation"]:
            config_data[term] = Config().items_to_dict(config_data[term])

        return config_data

    @staticmethod
    def store() -> None:
        """Saving the current run-time configuration to a file."""
        config_data = Config().to_dict()

        with open(Config.args.config, "w", encoding="utf8") as out:
            yaml.dump(config_data, out, default_flow_style=False)

    @staticmethod
    def present_params(element) -> None:
        """Presenting the configurations in a format style.

        Args:
            element (str): which part of params to present.
                        Default: all - present all of them
        i.e. :
            ********** ********** **********
            train:
                xxx: xxx:
                xxx: xxx:
            data:
                xxx: xxx:
                xxx: xxx:
        """

        empty_space = " " * 3
        configs_map = OrderedDict(
            [
                ("environment", Config().environment),
                ("data", Config().data),
                ("model", Config().model),
                ("train", Config().train),
                ("evaluation", Config().evaluation),
                ("logging", Config().logging),
            ]
        )

        def present_element(show_element):
            element_config = configs_map[show_element]
            element_config_dict = element_config._asdict()
            print(f"{show_element}: ")

            for item_name, item_value in element_config_dict.items():
                print(f"{empty_space} {item_name}: {item_value}")

        if element == "all":
            for term in list(configs_map.keys()):
                present_element(term)
        else:
            present_element(element)

    @staticmethod
    def make_basic_save_path() -> None:
        """ Make the saving path of different parts
            be the same., i.e.,:

            {model_name}__{data_name}__{grounding_model_name}__\
            {language_embedding_model_name}__{box_header_name}
        """

        method_name = Config.params["model_name"]
        data_name = Config.params["data_name"]
        grounding_module_name = Config.model.grounding_module_name
        language_module_name = Config.model.language_module_name
        header_name = Config.model.grounding_header_name

        if header_name is None:
            header_name = "None"

        target_name = "__".join(
            [
                method_name,
                data_name,
                grounding_module_name,
                language_module_name,
                header_name,
            ]
        )

        return target_name
