import os

import tensorflow as tf
from dpu_utils.utils import run_and_debug
import random
import numpy as np
import time
from tf2_gnn.cli_utils.training_utils import log_line, make_run_id, unwrap_tf_tracked_data, train
from typing import Dict, Any, Optional, Set, Type, List
from tf2_gnn.cli_utils.dataset_utils import get_dataset, get_model_file_path
from tf2_gnn.cli_utils.model_utils import get_model, load_weights_verbosely
import pickle
import json
from .data import DataFold, GraphDataset


class GraphObserverModel(): 
    def __init__(self, 
    model: str = 'RGCN', 
    task: str = 'NodeLevelRegression',
    random_seed:int = 0,  
    save_dir:str = os.path.dirname(os.getcwd())+'/trained-model', 
    max_epochs: int = 15, 
    patience: int = 5, 
    quiet: bool = False, 
    debug :bool = False, 
    load_saved_model: Optional[str] = None, 
    load_weights_only:bool = False, 
    run_test:bool = True, 
    azureml_logging:bool = False, 
    azure_info:str = "azure_auth.json", 
    data_params_override: Optional[str] = None, 
    model_params_override: Optional[str] = None, 
    hyperdrive_arg_parse:bool = False):
        self.model_str = model
        self.task_str = task
        self.max_epochs = max_epochs
        self.patience = patience
        self.quiet = quiet
        self.debug = debug
        self.save_dir = save_dir
        self.random_seed = random_seed
        self.load_saved_model = load_saved_model
        self.load_weights_only = load_weights_only
        self.run_test = run_test
        self.azureml_logging = azureml_logging
        self.data_params_override  = data_params_override
        self.model_params_override = model_params_override
        self.azure_info = azure_info
        self.hyperdrive_arg_parse = hyperdrive_arg_parse
        
    def __call__(self, raw_data: List[Dict[str, Any]]):
        self.dataset.load_data_from_list(raw_data, DataFold.TEST)
        load_weights_verbosely(self.trained_model_path, self.model)
        test_data = self.dataset.get_tensorflow_dataset(DataFold.TEST)
        predicted_targets, true_targets = self.model.predict(test_data)
        
        
    def fit(self, train_data: List[Dict[str, Any]], validation_data: List[Dict[str, Any]]): 
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
        tf.get_logger().setLevel("ERROR")

        run_and_debug(lambda: self._run_train(train_data, validation_data), debug)


    def _run_train(self, train_data:List[Dict[str, Any]], validation_data: List[Dict[str, Any]]): 
        os.makedirs(self.save_dir, exist_ok=True)
        run_id = make_run_id(self.model, self.task)
        log_file = os.path.join(self.save_dir, f"{run_id}.log")
        def log(msg):
            log_line(log_file, msg)

        log(f"Setting random seed {self.random_seed}.")
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)
        try:
            self.dataset, self.model = self.get_model_and_dataset(
                msg_passing_implementation=self.model,
                task_name=self.task,
                raw_graph_data = {"train": train_data, "validation": validation_data},  
                trained_model_file=self.load_saved_model,
                cli_data_hyperparameter_overrides=self.data_param_override,
                cli_model_hyperparameter_overrides=self.model_param_override,
                hyperdrive_hyperparameter_overrides=None,
                folds_to_load={DataFold.TRAIN, DataFold.VALIDATION},
                load_weights_only=self.load_weights_only,
                )
        except ValueError as err:
            print(err.args)
        log(f"Dataset parameters: {json.dumps(unwrap_tf_tracked_data(self.dataset._params))}")
        log(f"Model parameters: {json.dumps(unwrap_tf_tracked_data(self.model._params))}")

        if self.azureml_logging:
            from azureml.core.run import Run

            aml_run = Run.get_context()
        else:
            aml_run = None
        
        self.trained_model_path = train(
        self.model,
        self.dataset,
        log_fun=log,
        run_id=run_id,
        max_epochs=self.max_epochs,
        patience=self.patience,
        save_dir=self.save_dir,
        quiet=self.quiet,
        aml_run=aml_run,
        )


    def get_model_and_dataset(self,
        task_name: Optional[str],
        msg_passing_implementation: Optional[str],
        raw_graph_data: Dict[str,List[Dict[str, Any]]], 
        trained_model_file: Optional[str],
        cli_data_hyperparameter_overrides: Optional[str],
        cli_model_hyperparameter_overrides: Optional[str],
        hyperdrive_hyperparameter_overrides: Dict[str, str] = {},
        folds_to_load: Optional[Set[DataFold]] = None,
        load_weights_only: bool = False,): 
        if trained_model_file and not load_weights_only:
            with open(get_model_file_path(trained_model_file, "pkl"), "rb") as in_file:
                data_to_load = pickle.load(in_file)
            model_class = data_to_load["model_class"]
            dataset_class = data_to_load["dataset_class"]
            default_task_model_hypers = {}
   
        elif (trained_model_file and load_weights_only) or not trained_model_file:
            data_to_load = {}
            model_class, dataset_class = None, None

        # Load potential task-specific defaults:
            default_task_model_hypers = {}
            task_model_default_hypers_file = os.path.join(
                os.path.dirname(__file__),
                "default_hypers",
                "%s_%s.json" % (task_name, msg_passing_implementation),
            )
            print(
                f"Trying to load task/model-specific default parameters from {task_model_default_hypers_file} ... ",
                end="",
            )
            if os.path.exists(task_model_default_hypers_file):
                print("File found.")
                with open(task_model_default_hypers_file, "rt") as f:
                    default_task_model_hypers = json.load(f)
            else:
                print("File not found, using global defaults.")

            if not trained_model_file and load_weights_only:
                raise ValueError(
                    "Cannot load only weights when model file from which to load is not specified."
                )

        dataset = get_dataset(
            task_name,
            dataset_class,
            default_task_model_hypers.get("task_params", {}),
            data_to_load.get("dataset_params", {}),
            json.loads(cli_data_hyperparameter_overrides or "{}"),
            data_to_load.get("dataset_metadata", {}),
        )

        # Actually load data:
        if DataFold.TRAIN in folds_to_load: 
            print(f"Loading train data...")
            dataset.load_data_from_list(raw_graph_data["train"], DataFold.TRAIN)
        if DataFold.VALIDATION in folds_to_load: 
            print(f"Loading validation data...")
            dataset.load_data_from_list(raw_graph_data["validation"], DataFold.VALIDATION)
        if DataFold.TEST in folds_to_load: 
            print(f"Loading test data...")
            dataset.load_data_from_list(raw_graph_data["test"], DataFold.TEST)

        model = get_model(
            msg_passing_implementation,
            task_name,
            model_class,
            dataset,
            dataset_model_optimised_default_hyperparameters=default_task_model_hypers.get(
                "model_params", {}
            ),
            loaded_model_hyperparameters=data_to_load.get("model_params", {}),
            cli_model_hyperparameter_overrides=json.loads(
                cli_model_hyperparameter_overrides or "{}"
            ),
            hyperdrive_hyperparameter_overrides=hyperdrive_hyperparameter_overrides or {},
        )   

        data_description = dataset.get_batch_tf_data_description()
        model.build(data_description.batch_features_shapes)

        # If needed, load weights for model:
        if trained_model_file:
            print(f"Restoring model weights from {trained_model_file}.")
            load_weights_verbosely(trained_model_file, model)

        return dataset, model

