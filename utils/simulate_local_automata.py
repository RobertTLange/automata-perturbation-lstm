import numpy as np
import commentjson
from numba import jit
import multiprocessing as mp


def load_config(config_fname):
    """Load in a config JSON file and return as a dictionary"""
    json_config = commentjson.loads(open(config_fname, "r").read())
    return json_config


def generate_local_automata_dataset(
    sim_config, balanced_binary_labels=False, label_threshold=1, num_procs=None
):
    """Generate entire dataset of sequences."""
    if num_procs is None:
        num_procs = mp.cpu_count()
    # Detect number of CPUs and leave one untouched!
    offset = int(10 * sim_config["w0"] + 10 * sim_config["a0"])

    # Create a pool of workers, one for each fractal
    pool = mp.Pool(num_procs)
    rn = np.random.randint(0, 1000000, sim_config["N_sim"]) + offset

    # Set the number of perturbed cells based on the sample id
    stimuli = sim_config["cell_stimuli"]
    pertCells = []

    # Split sample into quarters of different perturbations
    if not balanced_binary_labels:
        split_ids = np.linspace(
            0, sim_config["N_sim"], len(stimuli) + 1
        ).astype(int)
        for sample_id in range(sim_config["N_sim"]):
            for i, s in enumerate(stimuli):
                if split_ids[i] <= sample_id < split_ids[i + 1]:
                    pertCells.append(s)

    # Split sample so that 0/1 labels are equally distributed
    else:
        cut_off = int(sim_config["N_sim"] / 2)
        pos_labels = [s for s in stimuli if s > label_threshold]
        pos_split_ids = np.linspace(0, cut_off, len(pos_labels) + 1).astype(int)
        neg_labels = [s for s in stimuli if s <= label_threshold]
        neg_split_ids = np.linspace(0, cut_off, len(neg_labels) + 1).astype(int)

        for pos_id in range(cut_off):
            for i, s in enumerate(pos_labels):
                if pos_split_ids[i] <= pos_id < pos_split_ids[i + 1]:
                    pertCells.append(s)

        for neg_id in range(cut_off):
            for i, s in enumerate(neg_labels):
                if neg_split_ids[i] <= neg_id < neg_split_ids[i + 1]:
                    pertCells.append(s)

    args_to_pass = [
        [sim_config, pertCells[i], rn[i]] for i in range(sim_config["N_sim"])
    ]
    all_sequences = pool.map(simulate_automata_sequence, args_to_pass)
    pool.close()

    # Stack all the collected traces on top of each other
    array_to_save = np.stack(all_sequences)

    # Save the data in .npy file or return array to process in memory
    if "base_save_fname" in sim_config.keys():
        ext_fname = (
            "_a_"
            + str(sim_config["a0"]).replace(".", "")
            + "_w_"
            + str(sim_config["w0"]).replace(".", "")
        )
        np.save(sim_config["base_save_fname"] + ext_fname, array_to_save)
    return array_to_save


def perform_perturbation(
    pertCells, q, swimTime1, perturb_connect, pz_size, sampled=None
):
    """Perform perturbation & resetting of state 1 occupancy time."""
    # 3 Types of perturbation - 1, 3, 5, 9 cells!
    Lx, Ly = q.shape
    XX = int(Lx / 2)
    YY = int(Ly / 2)
    new_q = q.copy()
    new_swimTime1 = swimTime1.copy()

    # Sample a set of connected cells to perturb
    if perturb_connect:
        new_q, new_swimTime1, sampled = perturb_connected(
            pertCells, new_q, new_swimTime1, XX, YY, sampled, pz_size
        )
    # Sample a set of random cells in perturbation zone
    else:
        new_q, new_swimTime1, sampled = perturb_random(
            pertCells, new_q, new_swimTime1, XX, YY, sampled, pz_size
        )
    return new_q, new_swimTime1, sampled


def perturb_connected(
    num_cells, new_q, new_swimTime1, XX, YY, sampled, pz_size=None
):
    """Perturb a randomly sampled & connected set of cells. For 3x3 case.
    -------------
    | 0 | 1 | 2 |
    | 3 | 4 | 5 |
    | 6 | 7 | 8 |
    -------------
    """
    # TODO: Generalize for more than 3x3 perturbation zone
    idx_x, idx_y = np.where(np.zeros((3, 3)) == 0)
    # List of cells with which each cell is connected
    # All cells which are one manhattan distance away
    connected = [
        [1, 3, 4],  # cell 0 is directly connected w. 1, 3, 4
        [0, 2, 3, 4, 5],
        [1, 4, 5],
        [0, 1, 4, 6, 7],
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 4, 7, 8],
        [3, 4, 7],
        [3, 4, 5, 6, 8],
        [4, 5, 7],
    ]

    # Cell start point to sample from
    sample_cells_from = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    # Only sample once at start of perturb sequence
    if sampled is None:
        sampled = []
        for i in range(num_cells):
            # Sample a random cell from all possible ones
            sampled.append(np.random.choice(sample_cells_from, 1)[0])
            # Reset all possible cells to sample & recalculate
            sample_cells_from = []
            for c in range(i + 1):
                sample_cells_from.extend(connected[sampled[c]])
            # Keep around only unique cell ids and once not already included
            sample_cells_from = list(set(sample_cells_from))
            sample_cells_from = [
                a for a in sample_cells_from if a not in sampled
            ]

    # Set cells to be active in raw activity array using cell id
    for c in sampled:
        new_q[XX - 1 + idx_x[c], YY - 1 + idx_y[c]] = 1
        new_swimTime1[XX - 1 + idx_x[c], YY - 1 + idx_y[c]] = 0.0
    return new_q, new_swimTime1, sampled


def perturb_random(num_cells, new_q, new_swimTime1, XX, YY, sampled, pz_size):
    """Perturb a randomly sampled set of cells. General pz_size case"""
    idx_x, idx_y = np.where(np.zeros((pz_size, pz_size)) == 0)

    # Only sample once at start of perturb sequence
    if sampled is None:
        possible_cell_ids = np.arange(pz_size * pz_size)
        sampled = np.random.choice(
            possible_cell_ids, size=num_cells, replace=False
        )

    # Set cells to be active in raw activity array using cell id
    for c in sampled:
        new_q[XX - 1 + idx_x[c], YY - 1 + idx_y[c]] = 1
        new_swimTime1[XX - 1 + idx_x[c], YY - 1 + idx_y[c]] = 0.0
    return new_q, new_swimTime1, sampled


@jit(nopython=True)
def one_step_euler_update(
    q, swimTime1, swimTime2, a0, w0, m, dt, t_act, t_ref, r
):
    """Calculate the spike rate & perform the update."""
    Lx, Ly = q.shape
    RateInd = np.zeros((Lx, Ly))

    # Generate random number for all cells before running loop
    new_q = q.copy()

    for id in range(Lx):
        for idd in range(Ly):
            indice1 = id - 1
            indice2 = id + 1
            indice3 = idd - 1
            indice4 = idd + 1

            if q[id][idd] == 0:
                # ONE STEP TO THE LEFT
                if 0 <= indice1 and indice1 < Lx:
                    if q[indice1][idd] == 1:
                        RateInd[id][idd] = RateInd[id][idd] + 1.0
                # ONE STEP TO THE RIGHT
                if 0 <= indice2 and indice2 < Lx:
                    if q[indice2][idd] == 1:
                        RateInd[id][idd] = RateInd[id][idd] + 1.0
                # ONE STEP DOWN
                if 0 <= indice3 and indice3 < Ly:
                    if q[id][indice3] == 1:
                        RateInd[id][idd] = RateInd[id][idd] + 1.0
                # ONE STEP UP
                if 0 <= indice4 and indice4 < Ly:
                    if q[id][indice4] == 1:
                        RateInd[id][idd] = RateInd[id][idd] + 1.0
                # ONE STEP TO THE LEFT & ONE STEP UP
                if 0 <= indice1 and indice1 < Lx:
                    if 0 <= indice4 and indice4 < Ly:
                        if q[indice1][indice4] == 1:
                            RateInd[id][idd] = RateInd[id][idd] + 1.0
                # ONE STEP TO THE LEFT & ONE STEP DOWN
                if 0 <= indice1 and indice1 < Lx:
                    if 0 <= indice3 and indice3 < Ly:
                        if q[indice1][indice3] == 1:
                            RateInd[id][idd] = RateInd[id][idd] + 1.0
                # ONE STEP TO THE RIGHT & ONE STEP UP
                if 0 <= indice2 and indice2 < Lx:
                    if 0 <= indice4 and indice4 < Ly:
                        if q[indice2][indice4] == 1:
                            RateInd[id][idd] = RateInd[id][idd] + 1.0
                # ONE STEP TO THE RIGHT & ONE STEP DOWN
                if 0 <= indice2 and indice2 < Lx:
                    if 0 <= indice3 and indice3 < Ly:
                        if q[indice2][indice3] == 1:
                            RateInd[id][idd] = RateInd[id][idd] + 1.0

            # The sigmoid - with sharpness = 1 - multiply exponent by m
            RateInd[id][idd] = ((RateInd[id][idd] / a0) ** (a0 * m)) / (
                1 + (RateInd[id][idd] / a0) ** (a0 * m)
            )
            RateInd[id][idd] = RateInd[id][idd] + w0

            # Perform Euler update for an individual cell
            if q[id][idd] == 0:
                if r[id][idd] < (RateInd[id][idd] * dt):
                    new_q[id][idd] = 1
                    swimTime1[id][idd] = 0.0

            if q[id][idd] == 1:
                swimTime1[id][idd] = swimTime1[id][idd] + dt
                if swimTime1[id][idd] >= t_act:
                    new_q[id][idd] = 2
                    swimTime2[id][idd] = 0.0

            if q[id][idd] == 2:
                swimTime2[id][idd] = swimTime2[id][idd] + dt
                if swimTime2[id][idd] >= t_ref:
                    new_q[id][idd] = 0
    return new_q, swimTime1, swimTime2


def simulate_automata_sequence(sim_input):
    """Simulate a single trajectory based on config."""
    # Unpack inputs for multiprocessing
    sim_config, pertCells, random_seed = (
        sim_input[0],
        sim_input[1],
        sim_input[2],
    )

    # Make sure that multiprocessing does not mess up random init
    np.random.seed(random_seed)

    # Get total number of cells & effective parameters of simulation
    N = sim_config["Lx"] * sim_config["Ly"]
    w0, a0, m = sim_config["w0"] / N, sim_config["a0"], sim_config["m"]
    t_act, t_ref = sim_config["t_act"], sim_config["t_ref"]

    # Initialize the placeholder arrays
    all_time_steps = []
    q = np.zeros((sim_config["Lx"], sim_config["Ly"]))

    # Clock counter for how long cells are in what state
    # Deterministic transitions only - see t_act/t_ref
    total_int_steps = int(sim_config["T_obs"] / sim_config["dt"])

    # Acticity measure (real) - percent active cells (state 1)
    activity_aux = 0.0
    save_activity = 0

    # Generate perturbation mask (don't track center 9 cells)
    q_pert = np.zeros((sim_config["Lx"], sim_config["Ly"]))
    XX = int(np.floor(sim_config["Lx"] / 2))
    YY = int(np.floor(sim_config["Ly"] / 2))
    offset = int(sim_config["pz_size"] / 2)
    q_pert[
        (XX - offset) : (XX + offset + 1), (YY - offset) : (YY + offset + 1)
    ] = 1

    # Uniformly initialize systems at beginning of sim.
    q = np.random.randint(3, size=(sim_config["Lx"], sim_config["Ly"]))
    swimTime1 = (
        np.random.rand(sim_config["Lx"], sim_config["Ly"]) * sim_config["t_act"]
    )
    swimTime2 = (
        np.random.rand(sim_config["Lx"], sim_config["Ly"]) * sim_config["t_ref"]
    )

    # If no timepoint of perturbation is explicitly provided - sample it!
    if sim_config["sample_perturb"]:
        perturb_range = np.arange(
            sim_config["perturb_range"][0], sim_config["perturb_range"][1], 1
        )
        perturb_time = np.random.choice(perturb_range)
    else:
        perturb_time = sim_config["T_perturb"]

    flag_perturb = 0
    perturb_duration = 0
    sampled = None
    # Run the forward euler integration
    for t in range(1, total_int_steps + 1):
        r = np.random.rand(sim_config["Lx"], sim_config["Ly"])
        # Perform perturbation of system when time point is reached
        if (
            (t * sim_config["dt"] == perturb_time) and flag_perturb == 0
        ) or flag_perturb == 1:
            q, swimTime1, sampled = perform_perturbation(
                pertCells,
                q,
                swimTime1,
                sim_config["perturb_connect"],
                sim_config["pz_size"],
                sampled,
            )
            flag_perturb = 1
            #  Keep pushing the system
            perturb_duration += sim_config["dt"]
            if perturb_duration >= sim_config["perturb_duration"]:
                flag_perturb = 2

        # Perform one step euler update
        new_q, swimTime1, swimTime2 = one_step_euler_update(
            q,
            swimTime1,
            swimTime2,
            a0,
            w0,
            m,
            sim_config["dt"],
            t_act,
            t_ref,
            r,
        )
        q = new_q.copy()

        # Only save every x-th timestep
        if t % sim_config["save_every_step"] == 0:
            # Calculate activity from all cells
            activityArray = q[q == 1]
            activity_aux = len(activityArray) / N
            refractoryArray = q[q == 2]
            refractory_aux = len(refractoryArray) / N
            save_activity += 1
            # Update stored array
            if sim_config["store_all_cells"]:
                qFlat = q[np.zeros((sim_config["Lx"], sim_config["Ly"])) <= 0]
            else:
                qFlat = q[q_pert <= 0]
            qFlat = np.append(qFlat, pertCells)
            qFlat = np.append(qFlat, activity_aux)
            qFlat = np.append(qFlat, refractory_aux)
            qFlat = np.append(qFlat, perturb_time)
            all_time_steps.append(qFlat)

    return np.array(all_time_steps)


if __name__ == "__main__":
    sim_config = {  # try k=2
        "pz_size": 6,  # 3 * k Perturbation zone size
        "perturb_connect": False,  # Perturb connected cells or random
        "a0": 2.7,
        "w0": 1.8,
        "m": 1.0,
        "Lx": 50,  # 25*k
        "Ly": 50,  # 25*k
        "T_obs": 40,  # 40*k? - total sim time
        "sample_perturb": 1,
        "perturb_range": [
            20,  # 20*4? - start perturb
            39,  # 39*4? - end perturb
        ],
        "fixed_perturb": 0,  # use fixed perturb timepoint
        "T_perturb": 30,  # fixed perturb timepoint
        "dt": 0.1,
        "N_sim": 500,
        "t_act": 1.0,
        "t_ref": 3.0,
        "save_every_step": 2,  # Downsample timeseries
        "store_all_cells": 1,
        "save_simulations": 0,
        "cell_stimuli": [
            0,
            4,  # 16, # 36/9 *1
            8,  # 32, # 36/9 *2
            12,  # 48, # 36/9 *3
            16,  # 64, # 36/9 *4
            20,  # 80, # 36/9 *5
            24,  # 96, # 36/9 *6
            28,  # 112, # 36/9 *7
            32,  # 128, # 36/9 *8
            36,  # 144 # 36/9 *9
        ],
        "perturb_duration": 3,
    }

    import argparse

    def get_sim_args():
        """Get env name, config file path & device to train from cmd line"""
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-a0", "--a0", type=float, default=2.7, help="a0 sim parameter"
        )
        parser.add_argument(
            "-w0", "--w0", type=float, default=1.7, help="w0 sim parameter"
        )
        return parser.parse_args()

    cmd_args = get_sim_args()
    sim_config["w0"] = cmd_args.w0
    sim_config["a0"] = cmd_args.a0
    # Simulate single sequence
    for c in [0, 4, 20, 36]:
        sim_input = [sim_config, c, 10]
        seq = simulate_automata_sequence(sim_input)
        print(seq.shape)
    print(sim_config["a0"], sim_config["w0"])
    out = generate_local_automata_dataset(sim_config)
    print(out.shape)
