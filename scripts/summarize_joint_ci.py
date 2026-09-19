import argparse
import math
import numpy as np
import pandas as pd

POPULATION_CRIT_95 = 21.350625538216956

def summarize(df):
    rows = []
    for n, g in df.groupby("n", sort=True):
        R = len(g)
        feasible = float(g["covered_95"].mean())
        mcse = math.sqrt(feasible * (1.0 - feasible) / R)
        T = np.sqrt(
            (np.sqrt(n) * g["alpha_hat"].to_numpy()) ** 2
            + (n ** 0.25 * g["x1_hat"].to_numpy()) ** 2
            + (n ** 0.25 * g["x2_hat"].to_numpy()) ** 2
        )
        oracle = float(np.mean(T <= POPULATION_CRIT_95))
        oracle_mcse = math.sqrt(oracle * (1.0 - oracle) / R)
        rows.append({
            "n": int(n),
            "R": R,
            "Bsim": int(g["Bsim"].iloc[0]),
            "feasible_coverage_95": feasible,
            "coverage_mcse": mcse,
            "oracle_limit_coverage_95": oracle,
            "oracle_coverage_mcse": oracle_mcse,
            "median_crit_radius_scaled": float(g["crit_radius_scaled"].median()),
            "population_crit_radius_scaled": POPULATION_CRIT_95,
            "median_full_width_alpha": float(g["full_width_alpha"].median()),
            "median_full_width_x": float(g["full_width_x"].median()),
            "boundary_hit_count": int(g["boundary_hit"].sum()),
            "boundary_hit_rate": float(g["boundary_hit"].mean()),
            "q99_max_abs_x": float(g["max_abs_x"].quantile(0.99)),
            "max_abs_x": float(g["max_abs_x"].max()),
            "empirical_T_q95": float(np.quantile(T, 0.95)),
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = summarize(pd.read_csv(a.input))
    out.to_csv(a.output, index=False)
    print(out.to_string(index=False))
