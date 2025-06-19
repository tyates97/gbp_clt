# External libraries
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
# from sklearn.metrics import mean_squared_error

# local modules
import distribution_management as dm
from optimisation import optimise_gaussian, optimise_q_gaussian

''' helper functions '''
def get_variables_to_plot(graph, max_subplots):
    k = min(max_subplots-3, len(graph.variables))
    indices = np.round(np.linspace(0, len(graph.variables)-1, k)).astype(int)
    variables_to_plot = [graph.variables[i] for i in indices]
    return variables_to_plot

def get_smoothing_factors_to_plot(graph, variables_to_plot):
    smoothing_factors_to_plot = []
    for smoothing_factor in [factor for factor in graph.factors if factor.factor_type == 'smoothing']:
        if smoothing_factor.neighbors[0] in variables_to_plot and len(smoothing_factors_to_plot) < 2:
            smoothing_factors_to_plot.append(smoothing_factor)
    return smoothing_factors_to_plot

def calculate_max_belief(graph):
    max_overall_val = 0.0
    for variable in graph.variables:
        if variable.belief is not None and len(variable.belief) > 0:
            max_variable_val = variable.belief.max()
            if max_variable_val > max_overall_val:
                max_overall_val = max_variable_val
    # Add a small margin for better visualization
    return max_overall_val * 1.1




''' plotting sub functions '''
def make_plot(x_range, y_range, x_title, y_title):        # plots a function in a new window
    plt.plot(x_range, y_range)
    plt.title(f'{x_title} vs {y_title}')
    plt.xlabel(f"{x_title}")
    plt.ylabel(f"{y_title}")
    plt.grid(True)
    plt.show()


def plot_smoothing_functions(fig, gs, smoothing_factors_to_plot, measurement_range):
    min_measurement = min(measurement_range)
    max_measurement = max(measurement_range)

    for i, factor in enumerate(smoothing_factors_to_plot):
        if i < 2:
            # Plot smoothing functions as a heatmap
            ax = fig.add_subplot(gs[0, i])
            im = ax.imshow(
                factor.function,
                origin='lower',
                cmap='viridis',
                extent=[min_measurement, max_measurement, min_measurement, max_measurement]
            )
            ax.set_title(f'Smoothing: {factor.name}')
            ax.set_xlabel(factor.neighbors[0].name)
            ax.set_ylabel(factor.neighbors[1].name)
            fig.colorbar(im, ax=ax, shrink=0.6)

def plot_prior_function(fig, gs, graph, measurement_range):
    # Plot prior functions
    prior_factors = [factor for factor in graph.factors if factor.factor_type == 'prior']
    ax = fig.add_subplot(gs[0, 2])      # TODO: fix hard coded position..
    ax.plot(measurement_range, prior_factors[0].function)
    ax.set_title(f'Prior: {prior_factors[0].name}')

# def plot_final_beliefs(fig, gs, variables_to_plot, measurement_range, comparison_distribution, gaussian_sigma, q, y_max, num_columns):
def plot_final_beliefs(fig, gs, variables_to_plot, measurement_range, comparison_distribution, y_max, num_columns):
    for i, var in enumerate(variables_to_plot):
        if var.belief is not None:
            ax = fig.add_subplot(gs[np.floor(i / (num_columns - 1)).astype(int) + 1, i % (num_columns - 1)])
            ax.plot(measurement_range, var.belief)

            if comparison_distribution == 'gaussian':
                # min_mse, optimal_sigma = optimise_gaussian(variable_to_optimise_for.belief, measurement_range)
                min_mse, gaussian_sigma = optimise_gaussian(var.belief, measurement_range)
                y_gauss = dm.create_gaussian_distribution(measurement_range, gaussian_sigma)
                ax.plot(measurement_range, y_gauss, color='green', label='Gaussian')
            if comparison_distribution == 'q gaussian':
                min_mse, q, gaussian_sigma = optimise_q_gaussian(var.belief, measurement_range)
                y_q_gauss = dm.create_q_gaussian_distribution(measurement_range, q, gaussian_sigma)
                ax.plot(measurement_range, y_q_gauss, color='green', label='Q-Gaussian')

            ax.set_title(f'{var.name} - MSE: {min_mse:.2e}')
            ax.set_ylim(0, y_max)
            ax.set_ylabel('Probability')

def plot_factor_graph(fig, gs, graph):
    factor_graph_ax = fig.add_subplot(gs[:,-1])
    factor_graph_ax.set_title("Factor Graph Structure")

    # Draw the factor graph
    pos = nx.spring_layout(graph.graph)  # Get a layout for the graph

    # Separate variable and factor nodes for coloring
    var_nodes = [variable.name for variable in graph.variables]
    factor_nodes = [factor.name for factor in graph.factors]

    # Draw nodes
    nx.draw_networkx_nodes(graph.graph, pos, nodelist=var_nodes, node_color='lightblue', node_size=1000, alpha=0.8,
                           label="Variables")
    nx.draw_networkx_nodes(graph.graph, pos, nodelist=factor_nodes, node_color='lightgreen', node_shape='s',
                           node_size=600, alpha=0.8, label="Factors")
    nx.draw_networkx_edges(graph.graph, pos, width=2.0, alpha=0.5)      # Draw edges
    nx.draw_networkx_labels(graph.graph, pos, font_size=15)             # Add labels to nodes
    factor_graph_ax.legend(labelspacing=1.5, borderpad=1.0)             # Add a legend
    factor_graph_ax.axis('off')                                         # Hide axis for the graph plot
    plt.tight_layout(pad=3.0, h_pad=2.0, rect=[0, 0, 1, 0.97])


''' main plotting function '''
def plot_results(graph, max_subplots, measurement_range, comparison_distribution):#, gaussian_sigma, q=1):
    # Define which variables/smoothing functions to plot & find max y-value
    variables_to_plot = get_variables_to_plot(graph, max_subplots)
    smoothing_factors_to_plot = get_smoothing_factors_to_plot(graph, variables_to_plot)
    y_max = calculate_max_belief(graph)

    # Set up the figure and layout
    fig = plt.figure(figsize=(20, 16))
    num_rows = min(np.round(np.sqrt(len(variables_to_plot))).astype(int) + 1,4)  # +1 to include space for the factor graph
    num_columns = min(np.ceil(np.sqrt(len(variables_to_plot))).astype(int) + 1, 4)
    gs = fig.add_gridspec(num_rows, num_columns, figure=fig, width_ratios=np.append(np.ones(num_columns - 1), [num_columns])) # make lots of space on right for the factor graph
    plt.suptitle("Belief Propagation on Factor Graph", fontsize=16)

    # Plot subplots
    plot_smoothing_functions(fig, gs, smoothing_factors_to_plot, measurement_range)
    plot_prior_function(fig, gs, graph, measurement_range)
    plot_final_beliefs(fig, gs, variables_to_plot, measurement_range, comparison_distribution, y_max, num_columns)
    # plot_final_beliefs(fig, gs, variables_to_plot, measurement_range, comparison_distribution, gaussian_sigma, q, y_max, num_columns)
    plot_factor_graph(fig, gs, graph)

    plt.show()