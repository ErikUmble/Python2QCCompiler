import asyncio
import logging

from sat_benchmark import CompileType, Trials
from sat_database import ThreeSatDB, ThreeSat

logging.disable()

# XAG stuff

sat_db = ThreeSatDB(db_name="random_3sat_problems.db")
sat_trials_db = Trials(db_name="3sat_trials.db")

import matplotlib.pyplot as plt
import numpy as np


def get_probability_data(trial_db, compile_type: CompileType):
    n_data = []
    grover_iterations_data = []
    probability_data = []

    for num_vars in range(3, 7):
        for grover_iterations in range(1, 2):
            print(f"(n, grover_iterations) = ({num_vars}, {grover_iterations})")
            trials = trial_db.get(
                num_vars=num_vars,
                grover_iterations=grover_iterations,
                compile_type=compile_type,
            )

            if len(trials) == 0:
                print(
                    f"Warning: no results for {num_vars} variables, complexity {grover_iterations}; skipping"
                )
                continue

            success_rates = np.zeros(len(trials))
            expected_success_rates = np.zeros(len(trials))

            for i, trial in enumerate(trials):
                print(trial.sat_instance)
                success_rates[i] = trial.success_rate
                expected_success_rates[i] = trial.expected_success_rate

            print(success_rates)
            print(expected_success_rates)
            n_data.append(num_vars)
            grover_iterations_data.append(grover_iterations)
            try:
                probability_data.append(np.mean(success_rates / expected_success_rates))
            except:
                print("ERROR")
            print(f"Mean success rate over expected: {probability_data[-1]}")

    return n_data, grover_iterations_data, probability_data


def plot_probability_data(
    n_data, grover_iterations_data, probability_data, title, filepath=None
):
    plt.figure(figsize=(20, 10))
    plt.scatter(
        n_data,
        grover_iterations_data,
        c=probability_data,
        cmap="RdYlGn",
        edgecolors="black",
        alpha=0.75,
        s=450,
    )
    plt.xticks(np.arange(min(n_data), max(n_data) + 1, 1))
    plt.yticks(
        np.arange(min(grover_iterations_data), max(grover_iterations_data) + 1, 1)
    )

    plt.xlabel("Verticies Count")
    plt.ylabel("Grover Iterations")
    plt.title(title)
    cbar = plt.colorbar()
    if filepath is not None:
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.show()


# --- Main execution logic ---
async def main():
    # run this cell to load the job results for all trials that are waiting pending job results
    print("Loading job results...")
    await sat_trials_db.load_results()
    print("Job results loaded.")

    print("\nGetting probability data...")
    n_data, grover_iterations_data, probability_data = get_probability_data(
        sat_trials_db,
        CompileType.CLASSICAL_FUNCTION,  # Make sure CompileType is defined/imported
    )
    print("Probability data collection complete.")

    # You can add plotting here if desired
    if n_data:  # Check if data was actually generated
        plot_title = "Probability Analysis for CLASSICAL_FUNCTION"
        plot_probability_data(
            n_data, grover_iterations_data, probability_data, plot_title
        )
    else:
        print("No data generated to plot.")


if __name__ == "__main__":
    # If running as a script
    asyncio.run(main())
    # If in a Jupyter Notebook, see Option 2 or 3 below.
