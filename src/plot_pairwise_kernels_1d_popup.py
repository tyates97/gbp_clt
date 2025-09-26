#!/usr/bin/env python3
"""
plot_pairwise_kernels_1d_popup.py
Pop-up matplotlib script to plot the *1D kernel/prior* that generated the first 3
pairwise (smoothing) factor functions.
Reversible: delete this file to revert.
"""
import sys
import numpy as np

# Prefer an interactive backend for pop-up windows
try:
    import matplotlib
    matplotlib.use("TkAgg")
except Exception:
    pass

import matplotlib.pyplot as plt

import config as cfg
from factor_graph import build_factor_graph
from belief_propagation import run_belief_propagation

def reconstruct_kernel_from_factor_matrix(F):
    """
    Reconstruct the 1D kernel 'prior' used to generate the Toeplitz-like 2D
    smoothing factor matrix F created by distribution_management.create_smoothing_factor_distribution.

    The original construction uses:
        diff = j - i
        idx  = diff + N//2
        F[i,j] = prior[-idx]   (NumPy negative indices wrap from the end)

    For each idx in [0..N-1], pick a valid (i,j) with diff = idx - N//2 and read F[i,j].
    Then map that back to prior at index (N - idx) % N (since prior[-idx] == prior[(N-idx)%N]).
    """
    N = F.shape[0]
    assert F.shape[0] == F.shape[1], "Factor matrix must be square"
    p = np.zeros(N, dtype=float)

    half = N // 2
    for idx in range(N):
        diff = idx - half  # j - i
        # choose a valid (i,j)
        if diff >= 0:
            i, j = 0, diff
        else:
            i, j = -diff, 0
        if 0 <= i < N and 0 <= j < N:
            val = F[i, j]
        else:
            # Fallback: clamp indices (shouldn't happen with logic above)
            i = max(0, min(N-1, i))
            j = max(0, min(N-1, j))
            val = F[i, j]
        # Map back to prior index: prior[-idx] == prior[(N-idx)%N]
        p[(N - idx) % N] = val

    # Normalise to compare on probability scale (optional)
    s = np.sum(p)
    if s > 0:
        p = p / s
    return p

def main():
    # --- set your target size here ---
    TARGET_W_PX, TARGET_H_PX = 632, 804      # same as the attached image
    DPI = 200                                 # pick the sharpness you want
    FIGSIZE = (TARGET_W_PX / DPI, TARGET_H_PX / DPI)
    
    # Build graph from current config
    graph = build_factor_graph(
        cfg.num_variables,
        cfg.num_priors,
        cfg.num_loops,
        cfg.graph_type,
        cfg.measurement_range,
        cfg.branching_factor,
        cfg.branching_probability,
        cfg.prior_location
    )

    # BP is not required but keeps state consistent with app usage
    run_belief_propagation(graph, getattr(cfg, "num_iterations", 1), getattr(cfg, "bp_pass_direction", "Both"))

    # First 3 pairwise/smoothing factors
    pairwise = [f for f in graph.factors if getattr(f, "factor_type", "smoothing") == "smoothing" and len(getattr(f, "neighbors", [])) == 2][:3]
    if not pairwise:
        print("No pairwise smoothing factors found.")
        sys.exit(0)

    # Prepare x-axis from measurement_range
    x = np.asarray(cfg.measurement_range, dtype=float)
    if x.ndim != 1 or len(x) != pairwise[0].function.shape[0]:
        # fallback to simple index axis
        x = np.arange(pairwise[0].function.shape[0])

    # Make a single figure with up to 3 line plots
    n = len(pairwise)
    fig, axes = plt.subplots(n, 1, figsize=(8, 2.8*n), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, fac in zip(axes, pairwise):
        kernel = reconstruct_kernel_from_factor_matrix(fac.function)
        # a, b = fac.neighbors
        fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
        ax.plot(x, kernel, color="tab:green")
        ax.set_title(None)
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        # remove ticks and tick labels
        ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

    # save exact pixel dimensions; each file has the same proportions
    out = f"kernel_{ax}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.show()
    fig.suptitle("Reconstructed 1D Kernels for First 3 Pairwise Factors", fontsize=12)
    plt.show()

if __name__ == "__main__":
    main()
