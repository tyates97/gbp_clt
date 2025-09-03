# External libraries
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import networkx as nx
# from sklearn.metrics import mean_squared_error

# local modules
import distribution_management as dm
from optimisation import optimise_gaussian

''' Setting the font sizes '''
plt.rcParams.update({
    'axes.titlesize': 8,
    'axes.labelsize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 8,
    'figure.titlesize': 10
})

''' helper functions '''
def get_variables_to_plot(graph, max_subplots):
    # # DEBUGGING: Choosing variables to plot
    # variables_to_plot = [graph.variables[18], graph.variables[24], graph.variables[30]]
    # k = min(max_subplots-6, len(graph.variables)) 
    # indices = np.round(np.linspace(0, len(graph.variables)-1, k)).astype(int)
    # for i in indices:
    #     variables_to_plot.append(graph.variables[i])

    k = min(max_subplots-3, len(graph.variables))
    indices = np.round(np.linspace(0, len(graph.variables)-1, k)).astype(int)
    variables_to_plot = [graph.variables[i] for i in indices]
    return variables_to_plot

def get_smoothing_factors_to_plot(graph, variables_to_plot):
    smoothing_factors_to_plot = []
    for smoothing_factor in [factor for factor in graph.factors if factor.factor_type == 'smoothing']:
        if smoothing_factor.neighbors[0] in variables_to_plot and len(smoothing_factors_to_plot) < 2:
        #DEBUGGING if smoothing_factor.name == 'fX5_X9' or smoothing_factor.name == 'fX9_X13':
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


def plot_final_beliefs(fig, gs, variables_to_plot, measurement_range, y_max, num_columns, show_comparison=True):
    for i, var in enumerate(variables_to_plot):
        if var.belief is not None:
            ax = fig.add_subplot(gs[np.floor(i / (num_columns - 1)).astype(int) + 1, i % (num_columns - 1)])
            ax.plot(measurement_range, var.belief)

            # min_mse, optimal_sigma = optimise_gaussian(variable_to_optimise_for.belief, measurement_range)
            min_mse, gaussian_sigma, gaussian_mu = optimise_gaussian(var.belief, measurement_range)
            y_gauss = dm.create_gaussian_distribution(measurement_range, gaussian_sigma, mu=gaussian_mu)
            if show_comparison:
                ax.plot(measurement_range, y_gauss, color='green', label='Gaussian')

            ax.set_title(f'{var.name} - MSE: {min_mse:.2e}')
            ax.set_ylim(0, y_max)
            ax.set_ylabel('Probability')


### Plotting the factor graph
def plot_factor_graph(fig, gs, graph):
    factor_graph_ax = fig.add_subplot(gs[:,-1])
    factor_graph_ax.set_title("Factor Graph Structure")

    if graph.is_grid:
        # Determine grid size (try to make it square)
        num_vars = len(graph.variables)
        grid_cols = int(np.ceil(np.sqrt(num_vars)))

        pos = {}
        # Place variable nodes in grid
        for idx, var in enumerate(graph.variables):
            row = idx // grid_cols
            col = idx % grid_cols
            pos[var.name] = (col, -row)  # y is negative so top row is at the top

        # Place factor nodes between their variable neighbors (average their positions)
        for factor in graph.factors:
            neighbor_names = [n.name for n in factor.neighbors]
            if factor.factor_type == 'smoothing':
                xs = [pos[n][0] for n in neighbor_names if n in pos]
                ys = [pos[n][1] for n in neighbor_names if n in pos]
                if xs and ys:
                    pos[factor.name] = (np.mean(xs), np.mean(ys))
            elif factor.factor_type == 'prior':
                # Place prior factors near the position of their single variable neighbor
                if len(neighbor_names) == 1 and neighbor_names[0] in pos:
                    pos[factor.name] = (pos[neighbor_names[0]][0]-0.3, pos[neighbor_names[0]][1]+0.3)  # Shift up and left a bit
                else:
                    pos[factor.name] = (0, 0)

    else:
        # Draw the factor graph
        pos = nx.spring_layout(graph.graph)  # Get a layout for the graph
    
    var_nodes = [v.name for v in graph.variables]
    smoothing_nodes = [f.name for f in graph.factors if f.factor_type == 'smoothing']
    prior_nodes = [f.name for f in graph.factors if f.factor_type == 'prior']

    # Draw nodes
    nx.draw_networkx_nodes(graph.graph, pos, nodelist=var_nodes, node_color='lightgreen', node_size=400, alpha=0.8,
                           label="Variables")
    nx.draw_networkx_nodes(graph.graph, pos, nodelist=smoothing_nodes, node_color='lightgrey', node_shape='s',
                           node_size=400, alpha=0.8, label="Smoothing factors")
    nx.draw_networkx_nodes(graph.graph, pos, nodelist=prior_nodes, node_color="#ffcccc", node_shape='s',
                           node_size=400, alpha=0.8, label="Prior factors")
    nx.draw_networkx_edges(graph.graph, pos, width=2.0, alpha=0.5)              # Draw edges
    var_labels = {name: name for name in var_nodes}
    nx.draw_networkx_labels(graph.graph, pos, labels=var_labels, font_size=10)  # Add labels to nodes
    # nx.draw_networkx_labels(graph.graph, pos, font_size=10)             
    factor_graph_ax.legend(labelspacing=2.5, borderpad=1.0)                     # Add a legend
    factor_graph_ax.axis('off')                                                 # Hide axis for the graph plot
    plt.tight_layout(pad=3.0, h_pad=2.0, rect=[0, 0, 1, 0.97])


### Plot a heatmap of how Gaussian each variables' belief is
def plot_heatmap(fig, gs, graph, measurement_range, cmap='viridis'):
    """
    Overlay a heatmap onto the factor-graph panel showing per-variable gaussian-fit MSE.
    Called after plot_factor_graph so it draws on the same right-hand axes (gs[:,-1]).
    """
    # compute per-variable MSEs
    var_nodes = [v.name for v in graph.variables]
    node_vals = []
    for v in graph.variables:
        if v.belief is None or len(v.belief) == 0:
            node_vals.append(np.nan)
        else:
            mse, _, _ = optimise_gaussian(v.belief, measurement_range)
            node_vals.append(mse)
    node_vals = np.array(node_vals, dtype=float)

    # prepare positions (same logic as plot_factor_graph)
    if graph.is_grid:
        num_vars = len(var_nodes)
        grid_cols = int(np.ceil(np.sqrt(num_vars)))
        pos = {}
        for i, name in enumerate(var_nodes):
            row = i // grid_cols
            col = i % grid_cols
            pos[name] = (col, -row)
    else:
        pos = nx.spring_layout(graph.graph)

    # Draw overlay on the factor-graph axis
    ax = fig.add_subplot(gs[:,-1])
    valid = ~np.isnan(node_vals)
    if np.any(valid):
        vmin = float(np.nanmin(node_vals))
        vmax = float(np.nanmax(node_vals))
    else:
        vmin, vmax = 0.0, 1.0

# use a green (Gaussian) -> red (non-Gaussian) colormap
    cmap_to_use = cmap if cmap is not None else 'RdYlGn_r'
    if cmap == 'viridis':  # default param case: prefer green->red mapping
        cmap_to_use = 'RdYlGn_r'
    nx.draw_networkx_nodes(graph.graph, pos, nodelist=var_nodes, node_color=node_vals,
                        cmap=cmap_to_use, vmin=vmin, vmax=vmax, node_size=400, alpha=0.95)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
    sm = plt.cm.ScalarMappable(cmap=cmap_to_use, norm=Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])  # required for colorbar
    fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04).set_label('Gaussian fit MSE')

    # # redraw labels for variables only (so labels remain readable)
    # var_labels = {name: name for name in var_nodes}
    # nx.draw_networkx_labels(graph.graph, pos, labels=var_labels, font_size=10, ax=ax)


''' main plotting function '''
def plot_results(graph, max_subplots, measurement_range, show_comparison=True, show_heatmap=False):
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
    plot_final_beliefs(fig, gs, variables_to_plot, measurement_range, y_max, num_columns, show_comparison)
    if show_heatmap:
        plot_heatmap(fig, gs, graph, measurement_range)
    else:
        plot_factor_graph(fig, gs, graph)
    return fig



''' Plot overlaid pdf distributions '''
def plot_volume_curves(pdf_volume, num_curves):
    
    print(f"Plotting {num_curves} PDFs...")
    # Get the dimensions of the pdf_volume
    height, width, max_disparity = pdf_volume.shape

    # Define a step size to select approximately your number of distributions
    step_y = int(height / int(np.sqrt(num_curves)))  
    step_x = int(width / int(np.sqrt(num_curves)))   

    # Create a new figure for the plot
    plt.figure(figsize=(12, 8))
    plt.title('100 Evenly-Spaced Disparity Probability Distributions')
    plt.xlabel('Disparity')
    plt.ylabel('Probability')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Iterate through the pdf_volume with the defined step sizes
    # This selects approximately 100 points
    for y in range(0, height, step_y):
        for x in range(0, width, step_x):
            # Extract the probability distribution for the current pixel
            pdf = pdf_volume[y, x, :]
            
            # Plot the distribution
            plt.plot(range(max_disparity), pdf, alpha=0.5)

    # Show the plot
    plt.show()


def plot_graph_beliefs(graph, num_curves, max_disparity):
    
    print(f"Plotting {num_curves} PDFs...")
    num_variables = len(graph.variables)
    
    # Define a step size to select approximately your number of distributions
    step = num_variables//num_curves

    # Create a new figure for the plot
    plt.figure(figsize=(12, 8))
    plt.title('100 Evenly-Spaced Disparity Probability Distributions')
    plt.xlabel('Disparity')
    plt.ylabel('Probability')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Iterate through the pdf_volume with the defined step sizes
    # This selects approximately 100 points
    for i, variable in enumerate(graph.variables):
        if i % step == 0:
            belief = variable.belief
            
            # Plot the distribution
            plt.plot(range(max_disparity), belief, alpha=0.5) 

    # Show the plot
    plt.show()


def plot_variance_heatmap(pdf_volume):
    # Get the dimensions of the pdf_volume
    height, width, max_disparity = pdf_volume.shape
    variance_vol = np.zeros((height, width))
    # Iterate through the pdf_volume with the defined step sizes
    # This selects approximately 100 points
    for y in range(0, height):
        for x in range(0, width):
            # Extract the probability distribution for the current pixel
            variance_vol[y,x] = np.var(pdf_volume[y,x,:])

    im = plt.imshow(variance_vol, cmap='RdYlGn') # Red (high variance) to Green (low variance)
    plt.colorbar(im, label='Var')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
