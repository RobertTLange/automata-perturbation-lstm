from utils.rollout_agent import (
    load_net_and_configs,
    get_rnn_predictions,
    generate_batch2predict,
)

from sklearn.metrics import roc_curve, roc_auc_score


def compute_roc_analysis(config_fname, chkpt_fname):
    """Load in checkpoint and perform forward passes for ROC."""
    rnn, train_config, sim_config = load_net_and_configs(
        config_fname, chkpt_fname
    )
    sim_config["N_sim"] = 1000
    sim_config["fixed_perturb"] = 1
    sim_config["perturb_duration"] = train_config["perturb_duration"]

    all_results = {"w0": sim_config["w0"], "a0": sim_config["a0"]}
    all_stim = [2, 3, 4, 5, 6, 7, 8, 9]
    for stim in all_stim:
        results = {"stim": stim}
        sim_config["cell_stimuli"] = [0, stim]
        # Test single prediction rollout
        X, y, _ = generate_batch2predict(sim_config, train_config)
        final_acc, preds, hiddens, probs = get_rnn_predictions(rnn, X, y)
        y_roc = y[:, -1]
        p_roc = probs[:, -1, 1]  # .data.cpu().numpy()
        fpr, tpr, thresholds = roc_curve(y_roc, p_roc, pos_label=1)
        score = roc_auc_score(y_roc, p_roc)
        results["fpr"] = fpr
        results["tpr"] = tpr
        results["auc"] = score
        all_results["stim_" + str(stim)] = results
        print(f"Stim: {stim} - ", results["auc"])
    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-config",
        "--config_fname",
        type=str,
        default="configs/local_critical.json",
        help="Path to configuration json.",
    )

    parser.add_argument(
        "-ckpt",
        "--ckpt_path",
        type=str,
        default="experiments/local_critical/models/final/final_seed_0.pt",
        help="Path to experiment checkpoint.",
    )
    args, _ = parser.parse_known_args()
    roc = compute_roc_analysis(args.config_fname, args.ckpt_path)
