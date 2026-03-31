import json
import numpy as np
from pathlib import Path

INPUT_PATH = Path("results/metrics_per_slice_test_all_models.json")
OUTPUT_PATH = Path("results/metrics_statistics_summary.json")


def compute_stats(values):

    values = np.array(values)

    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values))
    }


def main():

    print("Loading per-slice metrics...")

    with open(INPUT_PATH, "r") as f:
        data = json.load(f)

    print("Number of slices:", len(data))

    models = ["baseline", "cnn", "gan", "diffusion"]
    metrics = ["psnr", "ssim", "nmse"]

    results = {
        "num_slices": len(data)
    }

    for model in models:

        results[model] = {}

        for metric in metrics:

            values = [
                item[model][metric]
                for item in data
            ]

            results[model][metric] = compute_stats(values)

    print("\nComputed statistics:\n")

    for model in models:

        print(model)

        for metric in metrics:

            stats = results[model][metric]

            print(
                f"  {metric}: "
                f"{stats['mean']:.4f} ± {stats['std']:.4f}"
            )

        print()

    with open(OUTPUT_PATH, "w") as f:

        json.dump(results, f, indent=4)

    print("Saved statistics to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()