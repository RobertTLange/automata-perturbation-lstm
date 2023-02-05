import torch
import torch.nn as nn

# Import Deep Learning/RNN utilities
from core_rnn.criticality_rnn import CriticalityRNN
from core_rnn.rnn_dojo import RNNDojo
from core_rnn.optimizer import set_optimizer, set_lrate_schedule

# Import helpers for generating data.
from utils.local_data_gen import get_local_data
from utils.data_loader import get_train_test_split_loaders


def main(log, train_config, model_config):
    """Get the data ready & call the training script."""
    torch.set_num_threads(train_config.num_torch_threads)
    sim_config = train_config.sim_config.toDict()
    sim_config["a0"] = train_config.a0
    sim_config["w0"] = train_config.w0
    sim_config["perturb_duration"] = train_config.perturb_duration

    # Cut off window in beginning & reshape if desired
    balanced_binary_labels = (
        train_config.binary_labels and train_config.balanced_labels
    )
    print(sim_config)
    patches, raw = get_local_data(
        sim_config,
        train_config.start_t,
        train_config.binary_labels,
        balanced_binary_labels,
        train_config.one_hot_recode,
        train_config.label_threshold,
        train_config.num_patches_per_sequence,
        train_config.min_center_distance,
        train_config.max_center_distance,
    )
    X, y = patches[0], patches[1]
    in_shape = (
        X.shape[2] * X.shape[3]
        if not train_config.one_hot_recode
        else X.shape[2] * X.shape[3] * X.shape[4]
    )
    X = X.reshape(X.shape[0], X.shape[1], 1, in_shape)

    print("Shape of simulated data (X, y)", X.shape, y.shape)
    # Subselect corresponding to the train/test split
    train_loader, test_loader = get_train_test_split_loaders(
        X, y, train_config.train_test_split, train_config.batch_size
    )
    print(f"Train Data {len(train_loader)} - Test Data {len(test_loader)}")
    # Update the input dimension depending on the transformed data
    model_config["stem_info"]["input_dim"] = [1, X.shape[-1]]

    print("------- Start Training the RNN -------")
    train_rnn(log, train_config, model_config, train_loader, test_loader)
    return


def train_rnn(log, train_config, model_config, train_loader, test_loader):
    """Train the RNN Criticality Classifier."""
    # Set the training device
    device = torch.device(train_config.device_name)

    # Define the RNN architecture
    rnn = CriticalityRNN(**model_config).to(device)

    # Define the network optimizer - Base SGD!
    optimizer = set_optimizer(
        rnn,
        opt_type=train_config.opt_type,
        l_rate=train_config.l_rate,
        momentum=train_config.momentum,
        w_decay=train_config.w_decay,
    )

    if "lrate_schedule" in train_config.keys():
        scheduler = set_lrate_schedule(
            optimizer,
            train_config.lrate_schedule["schedule_type"],
            train_config.lrate_schedule["schedule_inputs"],
        )
    else:
        scheduler = None

    # Define the loss criterion
    criticality_criterion = nn.NLLLoss()
    # Instantiate the RNNDojo & start to train
    dojo = RNNDojo(
        rnn,
        optimizer,
        criticality_criterion,
        train_config.only_final_pred,
        device,
        problem_type="classification",
        train_loader=train_loader,
        test_loader=test_loader,
        train_log=log,
        log_batch_interval=train_config.log_every_batches,
        scheduler=scheduler,
        tboard_network_stats=train_config.log_network_stats,
        num_classes=model_config.num_classes,
    )
    dojo.train(num_epochs=train_config.num_epochs)


if __name__ == "__main__":
    # Import utilities for Cluster distributed runs
    try:
        from mle_toolbox import MLExperiment

        mle = MLExperiment(config_fname="configs/train/local.json")

        main(mle.log, mle.train_config, mle.model_config)
    # If not available load manually for single training run
    except Exception:
        import argparse
        from mle_logging import load_config, MLELogger

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-config",
            "--config_fname",
            type=str,
            default="configs/local.json",
            help="Path to configuration json.",
        )

        args, _ = parser.parse_known_args()
        config = load_config(args.config_fname, True)
        log = MLELogger(
            **config.log_config, experiment_dir="experiments/local/"
        )
        main(log, config.train_config, config.model_config)
