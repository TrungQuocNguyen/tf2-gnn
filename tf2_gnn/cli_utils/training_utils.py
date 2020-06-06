import json
import os
import random
import sys
import time
from typing import Dict, Optional, Callable, Any
import pickle
import datetime
import copy

import numpy as np
import tensorflow as tf
from tensorflow.python.training.tracking import data_structures as tf_data_structures
from dpu_utils.utils import RichPath

from ..data import DataFold, GraphDataset
from ..layers import get_known_message_passing_classes
from ..models import GraphTaskModel
from .model_utils import save_model, load_weights_verbosely, get_model_and_dataset
from .task_utils import get_known_tasks


def make_run_id(model_name: str, task_name: str, run_name: Optional[str] = None) -> str:
    """Choose a run ID, based on the --run-name parameter and the current time."""
    if run_name is not None:
        return run_name
    else:
        return "%s_%s__%s" % (model_name, task_name, time.strftime("%Y-%m-%d_%H-%M-%S"))


def log_line(log_file: str, msg: str):
    with open(log_file, "a") as log_fh:
        log_fh.write(msg + "\n")
    print(msg)


def train(
    model: GraphTaskModel,
    dataset: GraphDataset,
    log_fun: Callable[[str], None],
    run_id: str,
    max_epochs: int,
    patience: int,
    save_dir: str,
    quiet: bool = False,
    aml_run=None,
):
    train_data = dataset.get_tensorflow_dataset(DataFold.TRAIN).prefetch(3)
    valid_data = dataset.get_tensorflow_dataset(DataFold.VALIDATION).prefetch(3)

    save_file = os.path.join(save_dir, f"{run_id}_best.pkl")

    best_model = None
    best_valid_avg_loss, _, initial_valid_results = model.run_one_epoch(valid_data, training=False, quiet=quiet)
    best_valid_metric, best_val_str = model.compute_epoch_metrics(initial_valid_results)
    log_fun(f"Initial valid metric: {best_val_str}.")
    save_model(save_file, model, dataset)
    best_valid_epoch = 0

    #Set up summary writers to write the summaries to disk in a different logs directory (TensorBoard)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.getcwd() + '/logs/gradient_tape/' + current_time + '/train'
    valid_log_dir = os.getcwd() + '/logs/gradient_tape/' + current_time + '/valid'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)
    #Set up timer
    train_time_start = time.time()

    for epoch in range(1, max_epochs + 1):
        log_fun(f"== Epoch {epoch}")
        train_loss, train_speed, train_results = model.run_one_epoch(
            train_data, training=True, quiet=quiet
        )
        train_metric, train_metric_string = model.compute_epoch_metrics(train_results)
        log_fun(
            f" Train:  {train_loss:.4f} loss | {train_metric_string} | {train_speed:.2f} graphs/s",
        )
        valid_loss, valid_speed, valid_results = model.run_one_epoch(
            valid_data, training=False, quiet=quiet
        )
        valid_metric, valid_metric_string = model.compute_epoch_metrics(valid_results)
        log_fun(
            f" Valid:  {valid_loss:.4f} loss | {valid_metric_string} | {valid_speed:.2f} graphs/s",
        )


        if aml_run is not None:
            aml_run.log("task_train_metric", float(train_metric))
            aml_run.log("train_speed", float(train_speed))
            aml_run.log("task_valid_metric", float(valid_metric))
            aml_run.log("valid_speed", float(valid_speed))


        # log metrics for using later with TensorBoard 

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss, step=epoch)
        with valid_summary_writer.as_default():
            tf.summary.scalar('loss', valid_loss, step=epoch)

        # Save if good enough.
        if valid_loss < best_valid_avg_loss:
            log_fun(
                f"  (Best epoch so far, target metric decreased to {valid_loss:.5f} from {best_valid_avg_loss:.5f}.)",
            )
            save_model(save_file, model, dataset)
            best_valid_avg_loss = valid_loss
            best_valid_epoch = epoch
            best_model = model
        elif epoch - best_valid_epoch >= patience:
            total_time = time.time() - train_time_start
            log_fun(
                f"Stopping training after {patience} epochs without "
                f"improvement on validation metric.",
            )
            log_fun(f"Training took {total_time}s. Best validation metric: {best_valid_avg_loss}",)
            break
    log_fun(f"Running prediction of steering and acceleration of ego vehicles on train data ")
    train_predicted_targets, train_true_targets = best_model.predict(train_data)
    
    train_predicted_targets = train_predicted_targets.numpy()
    train_true_targets = train_true_targets.numpy()
    

    train_predicted_file = os.path.join(save_dir, f"train_predicted_targets.pickle")
    train_target_file = os.path.join(save_dir, f"train_true_targets.pickle")
    
    log_fun(f"Saving prediction to {train_predicted_file}")
    with open(train_predicted_file, 'wb') as handle:
        pickle.dump(train_predicted_targets, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()

    log_fun(f"Saving true target to {train_target_file}")
    with open(train_target_file, 'wb') as handle:
        pickle.dump(train_true_targets, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()
    #-----------------------------------------------------------------------------
    

    
    log_fun(f"Running prediction of steering and acceleration of ego vehicles on validation data ")
    valid_predicted_targets, valid_true_targets = best_model.predict(valid_data)
    valid_predicted_targets = valid_predicted_targets.numpy()
    valid_true_targets = valid_true_targets.numpy()

    valid_predicted_file = os.path.join(save_dir, f"valid_predicted_targets.pickle")
    valid_target_file = os.path.join(save_dir, f"valid_true_targets.pickle")
    
    log_fun(f"Saving prediction to {valid_predicted_file}")
    with open(valid_predicted_file, 'wb') as handle:
        pickle.dump(valid_predicted_targets, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()

    log_fun(f"Saving true target to {valid_target_file}")
    with open(valid_target_file, 'wb') as handle:
        pickle.dump(valid_true_targets, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()

    return save_file


def unwrap_tf_tracked_data(data: Any) -> Any:
    if isinstance(data, (tf_data_structures.ListWrapper, list)):
        return [unwrap_tf_tracked_data(e) for e in data]
    elif isinstance(data, (tf_data_structures._DictWrapper, dict)):
        return {k: unwrap_tf_tracked_data(v) for k, v in data.items()}
    else:
        return data


def run_train_from_args(args, hyperdrive_hyperparameter_overrides: Dict[str, str] = {}) -> None:
    # Get the housekeeping going and start logging:
    os.makedirs(args.save_dir, exist_ok=True)
    run_id = make_run_id(args.model, args.task)
    log_file = os.path.join(args.save_dir, f"{run_id}.log")
    
    def log(msg):
        log_line(log_file, msg)

    log(f"Setting random seed {args.random_seed}.")
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)

    data_path = RichPath.create(args.data_path, args.azure_info)
    try:
        dataset, model = get_model_and_dataset(
            msg_passing_implementation=args.model,
            task_name=args.task,
            data_path=data_path,
            trained_model_file=args.load_saved_model,
            cli_data_hyperparameter_overrides=args.data_param_override,
            cli_model_hyperparameter_overrides=args.model_param_override,
            hyperdrive_hyperparameter_overrides=hyperdrive_hyperparameter_overrides,
            folds_to_load={DataFold.TRAIN, DataFold.VALIDATION},
            load_weights_only=args.load_weights_only,
        )
    except ValueError as err:
        print(err.args)

    log(f"Dataset parameters: {json.dumps(unwrap_tf_tracked_data(dataset._params))}")
    log(f"Model parameters: {json.dumps(unwrap_tf_tracked_data(model._params))}")

    if args.azureml_logging:
        from azureml.core.run import Run

        aml_run = Run.get_context()
    else:
        aml_run = None

    trained_model_path = train(
        model,
        dataset,
        log_fun=log,
        run_id=run_id,
        max_epochs=args.max_epochs,
        patience=args.patience,
        save_dir=args.save_dir,
        quiet=args.quiet,
        aml_run=aml_run,
    )


    if args.run_test:
        data_path = RichPath.create(args.data_path, args.azure_info)
        log("== Running on test dataset")
        log(f"Loading data from {data_path}.")
        dataset.load_data(data_path, {DataFold.TEST})
        log(f"Restoring best model state from {trained_model_path}.")
        load_weights_verbosely(trained_model_path, model)
        test_data = dataset.get_tensorflow_dataset(DataFold.TEST)
        test_avg_loss, _, test_results = model.run_one_epoch(test_data, training=False, quiet=args.quiet)
        test_metric, test_metric_string = model.compute_epoch_metrics(test_results)
        log(f"Average epoch loss for test data is {test_avg_loss}")
        if aml_run is not None:
            aml_run.log("task_test_metric", float(test_metric))


def get_train_cli_arg_parser(default_model_type: Optional[str]=None):
    """
    Get an argparse argument parser object with common options for training
    GNN-based models.

    Args:
        default_model_type: If provided, the model type is downgraded from a
            positional parameter on the command line to an option with the
            given default value.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Train a GNN model.")
    # We use a somewhat horrible trick to support both
    #  train.py --model MODEL --task TASK --data_path DATA_PATH
    # as well as
    #  train.py model task data_path
    # The former is useful because of limitations in AzureML; the latter is nicer to type.
    if "--task" in sys.argv:
        model_param_name, task_param_name, data_path_param_name = "--model", "--task", "--data_path"
    else:
        model_param_name, task_param_name, data_path_param_name = "model", "task", "data_path"

    if default_model_type:
        model_param_name = "--model"
    parser.add_argument(
        model_param_name,
        type=str,
        choices=sorted(get_known_message_passing_classes()),
        default=default_model_type,
        help="GNN model type to train.",
    )
    parser.add_argument(
        task_param_name,
        type=str,
        choices=sorted(get_known_tasks()),
        help="Task to train model for.",
    )
    parser.add_argument(data_path_param_name, type=str, help="Directory containing the task data.")
    parser.add_argument(
        "--save-dir",
        dest="save_dir",
        type=str,
        default="trained_model",
        help="Path in which to store the trained model and log.",
    )
    parser.add_argument(
        "--model-params-override",
        dest="model_param_override",
        type=str,
        help="JSON dictionary overriding model hyperparameter values.",
    )
    parser.add_argument(
        "--data-params-override",
        dest="data_param_override",
        type=str,
        help="JSON dictionary overriding data hyperparameter values.",
    )
    parser.add_argument(
        "--max-epochs",
        dest="max_epochs",
        type=int,
        default=10000,
        help="Maximal number of epochs to train for.",
    )
    parser.add_argument(
        "--patience",
        dest="patience",
        type=int,
        default=25,
        help="Maximal number of epochs to continue training without improvement.",
    )
    parser.add_argument(
        "--seed", dest="random_seed", type=int, default=0, help="Random seed to use.",
    )
    parser.add_argument(
        "--run-name", dest="run_name", type=str, help="A human-readable name for this run.",
    )
    parser.add_argument(
        "--azure-info",
        dest="azure_info",
        type=str,
        default="azure_auth.json",
        help="Azure authentication information file (JSON).",
    )
    parser.add_argument(
        "--load-saved-model",
        dest="load_saved_model",
        help="Optional location to load initial model weights from. Should be model stored in earlier run.",
    )
    parser.add_argument(
        "--load-weights-only",
        dest="load_weights_only",
        action="store_true",
        help="Optional to only load the weights of the model rather than class and dataset for further training (used in fine-tuning on pretrained network). Should be model stored in earlier run.",
    )
    parser.add_argument(
        "--quiet", dest="quiet", action="store_true", help="Generate less output during training.",
    )
    parser.add_argument(
        "--run-test",
        dest="run_test",
        action="store_true",
        default=True,
        help="Run on testset after training.",
    )
    parser.add_argument(
        "--azureml_logging",
        dest="azureml_logging",
        action="store_true",
        help="Log task results using AML run context.",
    )
    parser.add_argument("--debug", dest="debug", action="store_true", help="Enable debug routines")

    parser.add_argument(
        "--hyperdrive-arg-parse",
        dest="hyperdrive_arg_parse",
        action="store_true",
        help='Enable hyperdrive argument parsing, in which unknown options "--key val" are interpreted as hyperparameter "key" with value "val".',
    )

    return parser
