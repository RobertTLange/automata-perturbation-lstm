from time import strftime, gmtime
import numpy as np
from numba import jit
from utils.simulate_local_automata import generate_local_automata_dataset


def one_hot_channel_recode(X):
    """Encode 0, 1, 2 (active vs. ref) states as separate channels."""
    all_samples = []
    for i in range(X.shape[0]):
        all_samples.append(one_hot_code(X[i]))
    return np.stack(all_samples)


def one_hot_code(x):
    """Encode the two stimuli separately."""
    channel_1 = np.zeros(x.shape)
    channel_2 = np.zeros(x.shape)
    channel_1[x == 1] = 1
    channel_2[x == 2] = 1
    stacked_channels = np.stack([channel_1, channel_2])
    # Reshape to (T, 2, L, L)
    return np.moveaxis(stacked_channels, 0, 1)


def get_local_data(
    sim_config,
    start_t=0,
    binary_labels=True,
    balanced_binary_labels=False,
    one_hot_recode=False,
    label_threshold=1,
    num_patches_per_sequence=5,
    min_center_distance=3,
    max_center_distance=5,
    num_procs=None,
    verbose=True,
):
    """Load in/generate local automate data & split into labels/features."""
    # Sample number of perturbed cells in range between 0 and 9
    # label_threshold determines how many cells need to be active for label 1
    effective_dt = sim_config["dt"] * sim_config["save_every_step"]
    now = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    if verbose:
        print(f"{now} - Begin with data")
    df = generate_local_automata_dataset(
        sim_config, balanced_binary_labels, label_threshold, num_procs
    )
    now = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    if verbose:
        print(f"{now} - Done with data")
    eff_size = sim_config["Lx"] * sim_config["Ly"]
    y_raw = df[:, int(np.round(start_t / effective_dt)) :, eff_size]
    perturb_time = df[:, int(np.round(start_t / effective_dt)) :, eff_size + 3][
        :, 0
    ]

    # Extract number of perturbed cells & change label to predict
    for i in range(perturb_time.shape[0]):
        y_raw[
            i,
            : (
                int(np.round(perturb_time[i] / effective_dt))
                - int(np.round(start_t / effective_dt))
            ),
        ] = 0

    label = y_raw.copy()

    # Change to 0/1 for whether there was a perturbation or not
    if binary_labels:
        label[y_raw <= label_threshold] = 0
        label[y_raw > label_threshold] = 1

    X = df[:, int(np.round(start_t / effective_dt)) :, :eff_size]
    # Shape: N_sim, T, Lx * Ly
    if verbose:
        print("X-raw", X.shape)
    # Here an additional 10GB of memory are occupied!
    X_2d = reshape_1d_2d(X, sim_config["Lx"])
    # Shape: N_sim, T, Lx, Ly
    if verbose:
        print("X-2D", X_2d.shape)

    patches = subsample_all_patches(
        X_2d,
        num_patches_per_sequence,
        min_center_distance=min_center_distance,
        max_center_distance=max_center_distance,
        L=sim_config["Lx"],
        pz_size=sim_config["pz_size"],
    )
    # Shape: N_sim, T, pz_size, pz_size
    if verbose:
        print("Patches", patches.shape)

    if one_hot_recode:
        # Separate channels for refractory and
        patches = one_hot_channel_recode(patches)
        # Shape: N_sim, T, 2, pz_size, pz_size
        if verbose:
            print("Patches-one-hot", patches.shape)

    label_patch = np.repeat(label, num_patches_per_sequence, axis=0)
    y_patch = np.repeat(y_raw, num_patches_per_sequence, axis=0)
    return (patches, label_patch, y_patch), (X_2d, label, y_raw)


def subsample_all_patches(
    X,
    num_patches_per_sequence,
    min_center_distance=3,
    max_center_distance=10,
    L=25,
    pz_size=3,
):
    """Loop over entire dataset and extract patches."""
    bs = X.shape[0]
    all_patches = []
    for i in range(bs):
        p, _, _ = subsample_patch(
            X[i],
            num_patches_per_sequence,
            min_center_distance,
            max_center_distance,
            L,
            pz_size,
        )
        all_patches.append(p)
    return np.concatenate(all_patches, axis=0)


def subsample_patch(
    X,
    num_patches_per_sequence,
    min_center_distance=3,
    max_center_distance=11,
    L=25,
    pz_size=3,
):
    """Sample patches from cellular automata timeseries."""
    XX = int(np.floor(L / 2))
    offset1, offset2 = int(np.floor(pz_size / 2)), int(np.ceil(pz_size / 2))
    center_begin, center_end = XX - offset1, XX + offset2
    # print(
    #     f"{XX}, {offset1}, {offset2}, Begin: {center_begin}, End: {center_end}"
    # )
    # Sample distance and direction away from the center cell
    dist_from_center = np.arange(min_center_distance, max_center_distance + 1)
    dir_away_from_center = np.arange(8)
    combs = np.array(
        np.meshgrid(dist_from_center, dir_away_from_center)
    ).T.reshape(-1, 2)
    dist_from_center, dir_away_from_center = combs[:, 0], combs[:, 1]
    # print(dist_from_center)
    patch_ids = np.random.choice(
        np.arange(len(combs)), num_patches_per_sequence, replace=False
    )

    patches = []
    sampled_dirs = []
    sampled_dists = []
    for p in patch_ids:
        sampled_dirs.append(dir_away_from_center[p])
        sampled_dists.append(dist_from_center[p])
        # 0 - left of center
        if dir_away_from_center[p] == 0:
            patches.append(
                X[
                    :,
                    (center_begin):(center_end),
                    (center_begin - dist_from_center[p]) : (
                        center_end - dist_from_center[p]
                    ),
                ]
            )
        # 1 - left + top
        elif dir_away_from_center[p] == 1:
            patches.append(
                X[
                    :,
                    (center_begin - dist_from_center[p]) : (
                        center_end - dist_from_center[p]
                    ),
                    (center_begin - dist_from_center[p]) : (
                        center_end - dist_from_center[p]
                    ),
                ]
            )
        # 2 - top
        elif dir_away_from_center[p] == 2:
            patches.append(
                X[
                    :,
                    (center_begin - dist_from_center[p]) : (
                        center_end - dist_from_center[p]
                    ),
                    (center_begin):(center_end),
                ]
            )
        # 3 - right + top
        elif dir_away_from_center[p] == 3:
            patches.append(
                X[
                    :,
                    (center_begin - dist_from_center[p]) : (
                        center_end - dist_from_center[p]
                    ),
                    (center_begin + dist_from_center[p]) : (
                        center_end + dist_from_center[p]
                    ),
                ]
            )
        # 4 - right
        elif dir_away_from_center[p] == 4:
            patches.append(
                X[
                    :,
                    (center_begin):(center_end),
                    (center_begin + dist_from_center[p]) : (
                        center_end + dist_from_center[p]
                    ),
                ]
            )
        # 5 - right + bottom
        elif dir_away_from_center[p] == 5:
            patches.append(
                X[
                    :,
                    (center_begin + dist_from_center[p]) : (
                        center_end + dist_from_center[p]
                    ),
                    (center_begin + dist_from_center[p]) : (
                        center_end + dist_from_center[p]
                    ),
                ]
            )
        # 6 - bottom
        elif dir_away_from_center[p] == 6:
            patches.append(
                X[
                    :,
                    (center_begin + dist_from_center[p]) : (
                        center_end + dist_from_center[p]
                    ),
                    (center_begin):(center_end),
                ]
            )
        # 7 - left + bottom
        elif dir_away_from_center[p] == 7:
            patches.append(
                X[
                    :,
                    (center_begin + dist_from_center[p]) : (
                        center_end + dist_from_center[p]
                    ),
                    (center_begin - dist_from_center[p]) : (
                        center_end - dist_from_center[p]
                    ),
                ]
            )
    # flat_patches = np.stack(patches).reshape(num_patches_per_sequence, X.shape[0], 9)
    return patches, sampled_dists, sampled_dirs


def reshape_1d_2d(X, L=25):
    """Reshape long vectors (619) back to 25x25 for conv (center blank)."""
    all_samples = []
    for i in range(X.shape[0]):
        all_samples.append(reshape_one_sample(X[i], L=L))
    return np.stack(all_samples)


@jit(nopython=True)
def reshape_one_sample(qflat, L=25):
    """Helper to reshape arrays to 2d w. 0s in center 9."""
    #  q_pert IS AN ARRAY TO MARK THE CELLS TO ERASE/Recover
    # q_pert = np.zeros((L, L))
    # XX = 12 # x-coordinate OF THE CENTER OF ARRAY
    # YY = 12 # y-coordinate OF THE CENTER OF ARRAY
    # q_pert[XX-1][YY-1] = 1
    # q_pert[XX-1][YY]   = 1
    # q_pert[XX-1][YY+1] = 1
    # q_pert[XX][YY-1]   = 1
    # q_pert[XX][YY]     = 1
    # q_pert[XX][YY+1]   = 1
    # q_pert[XX+1][YY-1] = 1
    # q_pert[XX+1][YY]   = 1
    # q_pert[XX+1][YY+1] = 1

    #  RESTORING THE ORIGINAL ARRAY
    T = qflat.shape[0]
    q_restored = np.zeros((T, L, L))

    for t in range(T):
        counter = 0
        for id in range(L):
            for idd in range(L):
                # if (q_pert[id][idd] < 1):
                q_restored[t][id][idd] = qflat[t][counter]
                counter = counter + 1
    return q_restored
