import torch
import numpy as np
from mle_logging import load_config
from core_rnn.criticality_rnn import CriticalityRNN, load_net_params
from utils.local_data_gen import get_local_data


def load_net_and_configs(config_fname, ckpth_fname):
    """Load a trained agent from checkpoint using config file."""
    # Sort out configs to return and build architecture from
    all_configs = load_config(config_fname, True)
    net_config = all_configs.model_config
    train_config = all_configs.train_config
    sim_config = dict(train_config.sim_config)

    # Define architecture and load trained weights
    device = torch.device("cpu")
    nn = CriticalityRNN(**net_config).to(device)
    load_net_params(nn, ckpth_fname)
    return nn, train_config, sim_config


def get_rnn_predictions(rnn, X, y):
    """Get prediction for batch of generated sequences."""
    # Put data on device & Initialize the hidden state
    device = torch.device("cpu")
    data, target = torch.Tensor(X).to(device), torch.Tensor(y).to(device)
    rnn.init_hidden(device=device, batch_size=data.size(0))

    # Perform a forward pass through the network
    all_preds, all_probs, all_hiddens = [], [], []
    for t in range(data.size(1)):
        data_t = data[:, t : t + 1, :]
        output = rnn(data_t.float())
        output = output.squeeze()
        probs = torch.exp(output)
        # Store predictions for specific timestep
        all_preds.append(probs.argmax(dim=1, keepdim=True))
        all_probs.append(probs)
        all_hiddens.append(rnn.rnn.hidden[0].squeeze())

    stacked_preds = torch.stack(all_preds, axis=1).squeeze()
    stacked_probs = torch.stack(all_probs, axis=1)
    stacked_hiddens = torch.stack(all_hiddens, axis=1)

    # If we train on a classification problem - change datatype!
    target = target.long()

    # Divide by batch dimension - Only final timestep prediction!
    overall_acc = stacked_preds.eq(
        target.view_as(stacked_preds)
    ).sum().item() / (target.size(0) * target.size(1))
    return (
        overall_acc,
        stacked_preds,
        stacked_hiddens,
        stacked_probs.detach().numpy(),
    )


def generate_batch2predict(sim_config, train_config, verbose=False):
    """Generate a batch to predict based on config."""
    balanced_binary_labels = (
        train_config.binary_labels and train_config.balanced_labels
    )
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
        verbose=verbose,
    )
    X, y, stim = patches[0], patches[1], patches[2]
    in_shape = (
        X.shape[2] * X.shape[3]
        if not train_config.one_hot_recode
        else X.shape[2] * X.shape[3] * X.shape[4]
    )
    X = X.reshape(X.shape[0], X.shape[1], in_shape)
    return X, y, stim


def get_psychometric_curve(
    rnn, train_config, sim_config, config_var, var_range
):
    """Loop over var list, construct predictions & return performance."""
    accuracies = []
    for var_value in var_range:
        if type(sim_config[config_var]) is list:
            sim_config[config_var] = [var_value]
        else:
            sim_config[config_var] = var_value
        X, y, stim = generate_batch2predict(sim_config, train_config)
        final_acc, preds, hiddens, probs = get_rnn_predictions(rnn, X, y)
        accuracies.append(final_acc)
    return np.array(accuracies)
