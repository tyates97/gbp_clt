import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import collections
import cv2

import optimisation as opt
import distribution_management as dm
import config as cfg
from factor_graph import build_factor_graph
from belief_propagation import run_belief_propagation
from graphics import plot_results


st.title("Interactive Factor Graph Belief Propagation")

# Sidebar controls
st.sidebar.title("Controls")
st.sidebar.subheader("Factor Graph Configuration")
cfg.num_variables = st.sidebar.slider("Number of Variables", 2, 200, 100)
cfg.graph_type = st.sidebar.selectbox("Graph Type",['Grid', 'Tree', 'Loopy'])
cfg.show_comparison = st.sidebar.checkbox("Show Gaussian best fit", value=True)
if cfg.graph_type == 'Tree':
    cfg.prior_location = st.sidebar.selectbox("Prior Location", ['root', 'leaf'])
elif cfg.graph_type == 'Grid':
    cfg.prior_location = st.sidebar.selectbox("Prior Location", ['corners', 'top', 'random'])
else:
    cfg.prior_location = 'root'
cfg.num_priors = st.sidebar.slider("Number of Priors", 1, cfg.num_variables, 30)

# Loopy Graphs submenu
if cfg.graph_type == 'Loopy':
    cfg.num_loops = st.sidebar.slider("Number of Loops", 1, 6, 3)

# Tree Graph submenu
if cfg.graph_type == 'Tree':
    # tree_prior_location = st.sidebar.selectbox("Prior Location",['root prior', 'leaf priors'])
    cfg.bp_pass_direction = st.sidebar.selectbox("Belief Propagation Direction",['Forward pass', 'Backward pass', 'Both'], index=2)
    cfg.branching_probability = st.sidebar.slider("Branching probability", 0.0, 1.0, 1.0, step=0.05)
    cfg.branching_factor = st.sidebar.slider("Branching Factor", 1, 7, 2, step=1)

# Grid Graph submenu
if cfg.graph_type == 'Grid':
    cfg.show_heatmap = st.sidebar.checkbox("Show Heatmap", value=False)
    cfg.sparse = True

# Additional variables
st.sidebar.subheader("Belief Propagation Configuration")
cfg.num_iterations = st.sidebar.slider("Number of BP Iterations", 1, 100, cfg.num_iterations)
cfg.belief_discretisation = st.sidebar.slider("Belief Discretisation", 8, 128, cfg.belief_discretisation, step=4)
cfg.prior_width = st.sidebar.slider("Prior Width", 4, int(cfg.belief_discretisation/2), cfg.prior_width, step=4)
cfg.smoothing_width = st.sidebar.slider("Smoothing Width", 4, int(cfg.belief_discretisation/2), cfg.smoothing_width, step=4)
cfg.random_seed = st.sidebar.number_input("Random Seed", value=42, step=1)
np.random.seed(cfg.random_seed)
cfg.rng = np.random.default_rng(seed=cfg.random_seed)  # Update RNG with user-defined seed


# Get disparity histogram
image_dir = 'data/stereo/teddy/'
left_ground_truth_filename = "disp2.png"
ground_truth = cv2.imread(image_dir+left_ground_truth_filename, cv2.IMREAD_GRAYSCALE)
ground_truth_signed = ground_truth.astype(np.int16)
all_diffs = dm.get_histogram_from_truth(ground_truth_signed)
hist, bin_edges = np.histogram(all_diffs, bins=2*cfg.belief_discretisation-1)
smoothing_kernel = dm.normalise(hist)

# Build and run
graph = build_factor_graph(
    cfg.num_variables,
    cfg.num_priors,
    cfg.num_loops,
    cfg.graph_type,
    cfg.measurement_range,
    cfg.prior_location,
    cfg.branching_factor,
    cfg.branching_probability
    # , hist=smoothing_kernel
)
graph = run_belief_propagation(graph, cfg.num_iterations)

# Plotting
st.subheader("Results")
fig = plot_results(
    graph,
    cfg.max_subplots,
    cfg.measurement_range,
    cfg.show_comparison,
    cfg.show_heatmap
)
st.pyplot(fig)


### Plotting MSE vs distance to prior

length_to_priors = list(opt.find_all_nearest_priors(graph).values())
# get all MSEs to best-fit gaussian
gauss_mse = []
for variable in graph.variables:
    min_mse,_,_ = opt.optimise_gaussian(variable.belief, cfg.measurement_range)
    gauss_mse.append(min_mse)

# Group MSEs by distance
mse_by_distance = collections.defaultdict(list)
for dist, mse in zip(length_to_priors, gauss_mse):
    mse_by_distance[dist].append(mse)

# Compute mean MSE for each distance
mean_mse = {dist: np.mean(mse_list) for dist, mse_list in mse_by_distance.items()}

# Sort by distance for plotting
sorted_distances = sorted(mean_mse.keys())
sorted_mse = [mean_mse[dist] for dist in sorted_distances]

# Create the plot
fig2, ax2 = plt.subplots()
ax2.plot(sorted_distances, sorted_mse, marker='o')
ax2.set_xlabel("Distance to nearest prior")
ax2.set_ylabel("MSE to best-fit Gaussian")
ax2.set_title("MSE vs. Distance to Prior")

# Show in a separate Streamlit section or expander
with st.expander("Show MSE vs. Distance to Prior plot"):
    st.pyplot(fig2)