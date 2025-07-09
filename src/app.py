import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from factor_graph import build_factor_graph
from belief_propagation import run_belief_propagation
from graphics import plot_results

st.title("Interactive Factor Graph Belief Propagation")

# Sidebar controls
num_variables = st.sidebar.slider("Number of Variables", 2, 20, 12)
num_loops = st.sidebar.slider("Number of Loops", 0, 6, 3)
num_priors = st.sidebar.slider("Number of Priors", 1, num_variables, 1)
is_tree = st.sidebar.checkbox("Tree factor graph", value=False)
branching_factor = st.sidebar.slider("If tree, Branching Factor", 1, 5, 2, step=1)
prior_distribution_type = st.sidebar.selectbox(
    "Prior Distribution Type",
    ['random', 'random symmetric', 'gaussian', 'top hat', 'horns', 'skew']
)

show_comparison = st.sidebar.checkbox("Show comparison distribution", value=True)

comparison_distribution = st.sidebar.selectbox(
    "Comparison Distribution",
    ['gaussian', 'q gaussian']
)
num_iterations = st.sidebar.slider("Number of BP Iterations", 1, 100, 50)
identical_smoothing_functions = st.sidebar.checkbox("Identical Smoothing Functions", value=False)
belief_discretisation = st.sidebar.slider("Belief Discretisation", 8, 128, 52, step=4)

min_measurement = -5
max_measurement = 5
max_subplots = 12
gauss_sigma = 2.3  # Temporary, in case needed for the definition of a Gaussian prior


measurement_range = np.linspace(min_measurement, max_measurement, belief_discretisation)

# Build and run
graph = build_factor_graph(
    num_variables,
    num_priors,
    num_loops,
    is_tree,
    identical_smoothing_functions,
    measurement_range,
    prior_distribution_type,
    gauss_sigma,
    branching_factor
)
graph = run_belief_propagation(graph, num_iterations)

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