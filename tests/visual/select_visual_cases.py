import json
import numpy as np
from pathlib import Path

METRICS_PATH = Path("results/metrics_per_slice_test_all_models.json")
OUTPUT_PATH = Path("results/selected_visual_cases.json")

UPPER_PSNR_PERCENTILE = 95
LOWER_PSNR_PERCENTILE = 1

MAX_SLICE = 5
TOP_K = 10


def collect_cases(indices, filtered):

    result = []

    for idx in indices:

        item = filtered[idx]

        result.append({
            "fname": item["fname"],
            "slice": item["slice"],
            "baseline_psnr": item["baseline"]["psnr"],
            "cnn_psnr": item["cnn"]["psnr"],
            "gan_psnr": item["gan"]["psnr"],
            "diffusion_psnr": item["diffusion"]["psnr"],
        })

    return result


def main():

    print("Loading metrics...")

    with open(METRICS_PATH, "r") as f:
        data = json.load(f)

    print("Total slices:", len(data))

    baseline_psnr = np.array([
        item["baseline"]["psnr"] for item in data
    ])

    upper_thr = np.percentile(baseline_psnr, UPPER_PSNR_PERCENTILE)
    lower_thr = np.percentile(baseline_psnr, LOWER_PSNR_PERCENTILE)

    filtered = [
        item for item in data
        if lower_thr <= item["baseline"]["psnr"] <= upper_thr
        and item["slice"] <= MAX_SLICE
    ]

    print("Slices after filtering:", len(filtered))
    print("PSNR range used:", lower_thr, "-", upper_thr)
    print("Slice range used: <=", MAX_SLICE)

    if len(filtered) == 0:
        raise RuntimeError("Filtering removed all slices.")

    cnn_psnr = np.array([item["cnn"]["psnr"] for item in filtered])
    gan_psnr = np.array([item["gan"]["psnr"] for item in filtered])
    diffusion_psnr = np.array([item["diffusion"]["psnr"] for item in filtered])

    mean_psnr = (cnn_psnr + gan_psnr + diffusion_psnr) / 3

    best_idx = np.argsort(diffusion_psnr)[-TOP_K:][::-1]

    hard_idx = np.argsort(cnn_psnr)[:TOP_K]

    median_psnr = np.median(mean_psnr)
    typical_idx = np.argsort(np.abs(mean_psnr - median_psnr))[:TOP_K]

    cnn_gan_diff = np.abs(cnn_psnr - gan_psnr)

    dis_idx = np.argsort(cnn_gan_diff)[-TOP_K:][::-1]

    selected = {
        "best_diffusion_cases": collect_cases(best_idx, filtered),
        "cnn_worst_cases": collect_cases(hard_idx, filtered),
        "typical_cases": collect_cases(typical_idx, filtered),
        "cnn_gan_disagreement_cases": collect_cases(dis_idx, filtered),
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(selected, f, indent=4)


    print("\nSelected cases:\n")

    for case_type, cases in selected.items():

        print("====", case_type, "====")

        for i, v in enumerate(cases):

            diff = abs(v["cnn_psnr"] - v["gan_psnr"])

            print(f"\nCandidate {i + 1}")
            print("file:", v["fname"])
            print("slice:", v["slice"])
            print("baseline PSNR:", v["baseline_psnr"])
            print("cnn PSNR:", v["cnn_psnr"])
            print("gan PSNR:", v["gan_psnr"])
            print("diffusion PSNR:", v["diffusion_psnr"])
            print("cnn-gan diff:", diff)

        print()

    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()