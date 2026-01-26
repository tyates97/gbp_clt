# recreate_bp_gbp_plot.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re






# DATA_DIR = Path("/home/tom/Projects/gbp_clt/exports/teddy")  # change if files are elsewhere
# OUT_FILE = "bp_gbp_runs_1_5_subplots.png"

# def load_series(name: str, run_id: int):
#     p = DATA_DIR / f"{name}_{run_id}.npy"
#     if not p.exists():
#         print(f"Missing: {p}")
#         return None
#     return np.load(p).astype(float)

# def plot_all_runs(run_ids=(1,2,3,4,5,6), cols=6, figsize=(16,4), out_file=OUT_FILE):
#     runs = list(run_ids)
#     n = len(runs)
#     rows = int(np.ceil(n / cols))
#     fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
#     axes = axes.flatten()

#     # load all series and compute common y-range
#     all_vals = []
#     series_list = []
#     for i, rid in enumerate(runs):
#         bp = load_series("bp_mse_values", rid)
#         gbp = load_series("gbp_mse_values", rid)
#         if bp is None and gbp is None:
#             series_list.append((None, None))
#             continue
#         if bp is None:
#             bp = np.full_like(gbp, np.nan)
#         if gbp is None:
#             gbp = np.full_like(bp, np.nan)
#         L = min(len(bp), len(gbp))
#         bp = bp[:L]
#         gbp = gbp[:L]
#         series_list.append((bp, gbp))
#         all_vals.append(bp)
#         all_vals.append(gbp)

#     # flatten and compute global y limits (ignore NaNs)
#     if all_vals:
#         stacked = np.concatenate([s.ravel() for s in all_vals if s.size > 0])
#         finite = stacked[np.isfinite(stacked)]
#         if finite.size > 0:
#             y_min = max(0.0, float(np.min(finite)))
#             y_max = float(np.max(finite))
#             y_top = y_max * 1.1 if y_max > 0 else 1.0
#         else:
#             y_min, y_top = 0.0, 1.0
#     else:
#         y_min, y_top = 0.0, 1.0

#     for i, rid in enumerate(runs):
#         ax = axes[i]
#         bp, gbp = series_list[i]
#         ax.set_title(f"Run {rid}")
#         if bp is None and gbp is None:
#             ax.text(0.5,0.5, "missing", ha="center", va="center")
#             ax.set_xticks([])
#             ax.set_yticks([])
#             continue
#         iters = np.arange(len(bp))
#         if np.any(np.isfinite(bp)):
#             ax.plot(iters, bp, color="#f0a500", lw=2, label="BP")
#         if np.any(np.isfinite(gbp)):
#             ax.plot(iters, gbp, color="#3b82f6", lw=2, label="GBP")
#         ax.set_xlim(0, iters[-1] if iters.size>0 else 1)
#         ax.set_ylim(y_min, y_top)
#         ax.grid(True, alpha=0.25)
#         if i == 0:
#             ax.legend(loc="upper right", framealpha=0.9)

#     # hide unused subplots
#     for j in range(len(runs), len(axes)):
#         axes[j].axis("off")

#     plt.tight_layout()
#     plt.savefig(out_file, dpi=300)
#     plt.show()
#     print("Saved:", out_file)

# if __name__ == "__main__":
#     plot_all_runs(run_ids=(1,2,4,5,6), cols=6, figsize=(20,4))







# --- config ---
DATA_DIR = Path("/home/tom/Projects/gbp_clt/exports/cones")  # change if files are elsewhere
# /home/tom/Projects/gbp_clt/exports/teddy/bp_mse_values_1.npy
BP_GLOB  = "bp_mse_values_*.npy"
GBP_GLOB = "gbp_mse_values_*.npy"
OUT_FILE = "bp_gbp_mse_mean_std.png"

# Colors to match your figure
COL_BP  = "#f0a500"   # orange
COL_GBP = "#6da5ff"   # blue
FILL_ALPHA = 0.25

def natural_key(p: Path):
    # sort ..._1.npy, ..._2.npy, ..._10.npy as humans expect
    m = re.search(r"(\d+)(?=\.npy$)", p.name)
    return int(m.group(1)) if m else 0

def load_runs(glob_pat):
    files = sorted(DATA_DIR.glob(glob_pat), key=natural_key)
    if not files:
        raise FileNotFoundError(f"No files matched: {DATA_DIR / glob_pat}")
    runs = [np.load(f) for f in files]
    # ensure same length across runs (truncate to the shortest, just in case)
    L = min(len(r) for r in runs)
    runs = np.stack([r[:L] for r in runs], axis=0)  # shape: (n_runs, n_iter)
    return runs, files

bp_runs, bp_files = load_runs(BP_GLOB)
gbp_runs, gbp_files = load_runs(GBP_GLOB)

# iterations (assumes index 0..L-1 corresponds to iteration)
iters = np.arange(bp_runs.shape[1])

# aggregates
def stats(runs):
    return runs.mean(axis=0), runs.std(axis=0)

a = np.zeros((10,2001))
a[:5,:] = gbp_runs
a[5:,:] = gbp_runs
bp_mean, bp_std = stats(bp_runs)
gbp_mean, gbp_std = stats(a)

# --- plot ---
plt.figure(figsize=(6, 7))

# font size settings (edit values here)
LABEL_FONT = 18
TICK_FONT = 18
LEGEND_FONT = 16
LINEWIDTH = 3

# BP
plt.plot(iters, bp_mean, lw=3, color=COL_BP, label=f"BP mean (n={bp_runs.shape[0]})")
plt.fill_between(iters, bp_mean - bp_std, bp_mean + bp_std,
                 color=COL_BP, alpha=FILL_ALPHA, label="BP ±1 std")
# GBP
plt.plot(iters, gbp_mean, lw=3, color=COL_GBP, label=f"GBP mean (n={gbp_runs.shape[0]})")
plt.fill_between(iters, gbp_mean - gbp_std, gbp_mean + gbp_std,
                 color=COL_GBP, alpha=FILL_ALPHA, label="GBP ±1 std")

# cosmetics to mirror your image
plt.xlabel("Iterations", fontsize=LABEL_FONT)
plt.ylabel("MSE", fontsize=LABEL_FONT)

# Force y-axis to start at 0 and set top a little above the largest plotted value
max_bp = np.nanmax(bp_mean + bp_std)
max_gbp = np.nanmax(gbp_mean + gbp_std)
y_top = max(max_bp, max_gbp)
if not np.isfinite(y_top) or y_top <= 0:
    y_top = 1.0
else:
    y_top = y_top * 1.1
plt.ylim(0.0, y_top)

# x-axis ticks every 500 iterations (include final iteration)
step = 500
last = int(iters[-1]) if iters.size else 0
xticks = np.arange(0, last + step, step)
if xticks[-1] != last:
    xticks = np.append(xticks, last)
plt.xticks(xticks, fontsize=TICK_FONT)
plt.yticks(fontsize=TICK_FONT)

plt.xlim(0, last)
plt.grid(True, alpha=0.25)

plt.legend(frameon=True, fancybox=True, framealpha=0.9, loc="upper right", fontsize=LEGEND_FONT)

plt.tight_layout()
plt.savefig(OUT_FILE, dpi=300)
plt.show()

print("Saved plot to:", OUT_FILE)
print("Loaded BP files:", [p.name for p in bp_files])
print("Loaded GBP files:", [p.name for p in gbp_files])
