"""
Reading runtime parameters from a configuration file.
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


logging.getLogger("httpx").setLevel(logging.CRITICAL)


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
    def construct_join(loader: Loader, node: yaml.Node) -> Any:
        """Support os.path.join at node."""
        seq = loader.construct_sequence(node)
        return "/".join([str(i) for i in seq])

    @staticmethod
    def construct_include(loader: Loader, node: yaml.Node) -> Any:
        """Include file referenced at node."""
        filename = os.path.abspath(
            os.path.join(loader.root_path, loader.construct_scalar(node))
        )
        extension = os.path.splitext(filename)[1].lstrip(".")

        extension = os.path.splitext(filename)[1].lstrip(".")
        with open(filename, "r", encoding="utf-8") as config_file:
            if extension in ("yaml", "yml"):
                return yaml.load(config_file, Loader)
            elif extension in ("json",):
                return json.load(config_file)
            else:
                return "".join(config_file.readlines())

    @staticmethod
    def construct_merge_include(loader: Loader, node: yaml.Node) -> Any:
        """
        Include multiple files referenced at node with tag `!minclude`.

        For merging multiple files under one include, the format of should be
        <>;;<>;;
        where <> will be the path of one config file while `;;` is used to
        separate the files
        Lead to: !minclude: <>;;<>;;
        """
        # load the files as a whole
        to_be_include = loader.construct_scalar(node)
        include_files = to_be_include.split(";;")

        merged_content = {}
        for file in include_files:
            filepath = os.path.abspath(os.path.join(loader.root_path, file))

            extension = os.path.splitext(filepath)[1].lstrip(".")
            file_content = {}
            with open(filepath, "r", encoding="utf-8") as config_file:
                if extension in ("yaml", "yml"):
                    file_content = yaml.load(config_file, Loader)
                elif extension in ("json",):
                    file_content = json.load(config_file)
                else:
                    file_content = "".join(config_file.readlines())
            merged_content.update(file_content)

        return merged_content

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

            # add the tag to support the !include
            yaml.add_constructor("!include", Config.construct_include, Loader)
            yaml.add_constructor("!minclude", Config.construct_merge_include, Loader)
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
            Config.trainer = Config.namedtuple_from_dict(config["trainer"])
            Config.evaluation = Config.namedtuple_from_dict(config["evaluation"])
            Config.logging = Config.namedtuple_from_dict(config["logging"])

            HOMEPATH = str(Path.home())

            # Customizable dictionary of global parameters
            Config.params: dict = {}

            # A run ID is unique in an experiment
            # The precisions are unique for all parts of the model
            Config.params["run_id"] = os.getpid()

            # Get the prefix of the filename
            # It is hope that the prefix will be the method name
            method_name = os.path.basename(filename).split("_")[0]
            Config.params["method_name"] = method_name

            # Project dir
            # The base path used for all datasets, models, checkpoints, and results
            Config.params["base_path"] = Config.args.base.replace("~", HOMEPATH)

            project_path = Config.params["base_path"]

            # Get the name of the model
            print(Config().model)
            Config.params["model_name"] = Config().model.model_name

            # Get the name of the data
            Config.params["data_name"] = Config().data.data_name

            # the basic saving dir name in
            # models/checkpoints/results/logging
            basic_path_name = Config.create_save_name()

            # data dir
            # we hope the data can be placed under the project dir directly
            #   instead of be placing under project_path/project_name
            data_path = Config().data.data_path.replace("~", HOMEPATH)

            if data_path is not None:
                # if the set dir is not correct, we redirect to
                #   the default dir {data_path}/data_path
                data_path = Config().data.data_path.replace("~", HOMEPATH)
                if not os.path.exists(data_path):
                    expected_data_path = os.path.join(project_path, data_path)

                    Config.data = Config.data._replace(
                        data_path=Path(expected_data_path).as_posix()
                    )

            # now = datetime.datetime.now()
            # now_time_str = (
            #     now.strftime("%Y-%m-%d %H:%M:%S").replace(" ", "-").replace(":", "-")
            # )
            # Logging dir
            # The 'experiment_path' should be set in the path defined by -b
            # as the project_path will be utilized as the experiment_path
            # directly
            experiment_path = Config().logging.experiment_path.replace("~", HOMEPATH)
            if not os.path.exists(experiment_path):
                desired_path = Path(project_path).as_posix()
                Config.logging = Config.logging._replace(experiment_path=desired_path)

            # Resume checkpoint
            data_checkpoint_path = os.path.join(
                Config().logging.checkpoint_path, basic_path_name
            ).replace("~", HOMEPATH)
            if not os.path.exists(data_checkpoint_path):
                Config.logging = Config.logging._replace(
                    checkpoint_path=Path(
                        os.path.join(
                            Config().logging.experiment_path,
                            "checkpoints",
                            basic_path_name,
                        )
                    ).as_posix()
                )
            else:
                Config.logging = Config.logging._replace(
                    checkpoint_path=data_checkpoint_path
                )

            data_result_path = os.path.join(
                Config().logging.result_path, basic_path_name
            ).replace("~", HOMEPATH)
            if not os.path.exists(data_result_path):
                Config.logging = Config.logging._replace(
                    result_path=Path(
                        os.path.join(
                            Config().logging.experiment_path, "results", basic_path_name
                        )
                    ).as_posix()
                )
            else:
                Config.logging = Config.logging._replace(result_path=data_result_path)

            # logging
            logging_path = os.path.join(
                Config().logging.logging_path, basic_path_name
            ).replace("~", HOMEPATH)
            if not os.path.exists(logging_path):
                Config.logging = Config.logging._replace(
                    logging_path=Path(
                        os.path.join(
                            Config().logging.experiment_path,
                            "logging",
                            basic_path_name,
                        )
                    ).as_posix()
                )
            else:
                Config.logging = Config.logging._replace(logging_path=logging_path)

            # Visual results
            data_visualization_path = os.path.join(
                Config().logging.visualization_path, basic_path_name
            ).replace("~", HOMEPATH)
            if not os.path.exists(data_visualization_path):
                Config.logging = Config.logging._replace(
                    visualization_path=Path(
                        os.path.join(
                            Config().logging.experiment_path,
                            "visualizations",
                            basic_path_name,
                        )
                    ).as_posix()
                )
            else:
                Config.logging = Config.logging._replace(
                    visualization_path=data_visualization_path
                )

            # Pretrained models
            if not os.path.exists(Config().model.pretrained_models_path):
                Config.model = Config.model._replace(
                    pretrained_models_path=Path(
                        os.path.join(
                            Config().logging.experiment_path, "pretrained_models"
                        )
                    ).as_posix()
                )

            os.makedirs(Config().logging.experiment_path, exist_ok=True)
            os.makedirs(Config().logging.checkpoint_path, exist_ok=True)
            os.makedirs(Config().logging.result_path, exist_ok=True)
            os.makedirs(Config().logging.visualization_path, exist_ok=True)
            os.makedirs(Config().logging.logging_path, exist_ok=True)
            os.makedirs(Config().model.pretrained_models_path, exist_ok=True)

            # Saving the given config file to the corresponding
            # results/models/checkpoints
            config_file_name = os.path.basename(filename)
            config_log_path = os.path.join(
                Config().logging.logging_path, config_file_name
            )
            config_checkpoint_path = os.path.join(
                Config().logging.checkpoint_path, config_file_name
            )
            config_result_path = os.path.join(
                Config().logging.result_path, config_file_name
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
                    Config().logging.logging_path, file_log_type + "_" + log_name
                )

                file_handler = logging.FileHandler(log_file_name)
                file_handler.setLevel(file_log_numeric_level)
                file_handler.setFormatter(formatter)

                root_logger.addHandler(file_handler)

        return cls._instance

    @staticmethod
    def namedtuple_from_dict(obj):
        """Create a named tuple from a dictionary."""
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
        """Return the device to be used for training."""
        device = "cpu"

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            if hasattr(Config().environment, "device_id"):
                device = "cuda:" + str(Config().environment.device_id)
            else:
                device = "cuda:" + str(np.random.randint(0, torch.cuda.device_count()))
        if torch.backends.mps.is_available():
            device = "mps"

        return device

    @staticmethod
    def items_to_dict(base_dict) -> dict:
        """
        Convert items of the dict to be dict if possible.

        The main purpose of this function is to address the
        condition of nested Config term, such as
        {"key1": Config(k11=5, k12=7),
         "key2": Config(k12=5, k22=Config(m=1, n=7))}
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
        """Convert the current run-time configuration to a dict."""

        config_data = dict()
        config_data["trainer"] = Config.trainer._asdict()
        config_data["environment"] = Config.environment._asdict()
        config_data["data"] = Config.data._asdict()
        config_data["model"] = Config.model._asdict()
        config_data["logging"] = Config.logging._asdict()
        config_data["evaluation"] = Config.evaluation._asdict()
        for term in [
            "trainer",
            "environment",
            "data",
            "model",
            "logging",
            "evaluation",
        ]:
            config_data[term] = Config().items_to_dict(config_data[term])

        return config_data

    @staticmethod
    def store() -> None:
        """Save the current run-time configuration to a file."""
        config_data = Config().to_dict()

        with open(Config.args.config, "w", encoding="utf8") as out:
            yaml.dump(config_data, out, default_flow_style=False)

    @staticmethod
    def create_save_name() -> None:
        """
        Create the basic save name for the current run-time configuration.
        """
        method_name = Config.params["method_name"]
        show_name = (
            f"{method_name}Reasoning" if "Reasoning" not in method_name else method_name
        )
        save_name = "__".join(
            [
                show_name,
                Config.params["model_name"],
                Config.model.prompt_type,
                Config.params["data_name"],
            ]
        )

        return save_name
