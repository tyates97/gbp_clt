import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from factor_graph import build_factor_graph
from belief_propagation import run_belief_propagation
from graphics import plot_results

st.title("Interactive Factor Graph Belief Propagation")

# Sidebar controls
st.sidebar.title("Controls")
st.sidebar.subheader("Factor Graph Configuration")
num_variables = st.sidebar.slider("Number of Variables", 2, 50, 15)
prior_distribution_type = st.sidebar.selectbox("Prior Distribution Type",['random', 'random symmetric', 'gaussian', 'top hat', 'horns', 'skew'])
graph_type = st.sidebar.selectbox("Graph Type",['Loopy Graph', 'Tree Graph'])
identical_smoothing_functions = st.sidebar.checkbox("Identical Smoothing Functions", value=False)
show_comparison = st.sidebar.checkbox("Show Gaussian best fit", value=True)

# Loopy Graphs Topology
st.sidebar.subheader("Loopy Graphs")
num_loops = st.sidebar.slider("Number of Loops", 1, 6, 3)
num_priors = st.sidebar.slider("Number of Priors", 1, num_variables, 1)

# Tree Graph Topology
st.sidebar.subheader("Tree Graphs")
bp_pass_direction = st.sidebar.selectbox("Belief Propagation Direction",['Forward pass', 'Backward pass', 'Both'])
branching_factor = st.sidebar.slider("Branching Factor", 1, 5, 2, step=1)
branching_probability = st.sidebar.slider("Branching probability", 0.0, 1.0, 1.0, step=0.05)
tree_prior_location = st.sidebar.selectbox("Prior Location",['root prior', 'leaf priors'])


# Additional variables
st.sidebar.subheader("Belief Propagation Configuration")
num_iterations = st.sidebar.slider("Number of BP Iterations", 1, 100, 50)
belief_discretisation = st.sidebar.slider("Belief Discretisation", 8, 128, 52, step=4)

comparison_distribution = 'gaussian'
min_measurement = -5
max_measurement = 5
max_subplots = 12
measurement_range = np.linspace(min_measurement, max_measurement, belief_discretisation)

# Build and run
graph = build_factor_graph(
    num_variables,
    num_priors,
    num_loops,
    graph_type,
    identical_smoothing_functions,
    measurement_range,
    prior_distribution_type,
    branching_factor,
    branching_probability,
    tree_prior_location
)
graph = run_belief_propagation(graph, num_iterations, bp_pass_direction)

# Plotting
st.subheader("Results")
fig = plot_results(
    graph,
    max_subplots,
    measurement_range,
    comparison_distribution,
    show_comparison
)
st.pyplot(fig)