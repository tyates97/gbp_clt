"""
factor_graph.py
"""
# External libraries
import networkx as nx
import numpy as np

# Local modules
import distribution_management as dm
import config as cfg


''' classes '''
class VariableNode:
    def __init__(self, name, belief_discretisation):
        self.name = name
        self.belief = dm.normalise(np.ones(belief_discretisation)) # Initialised belief
        self.neighbors = []

    def add_neighbor(self, factor_node):
        self.neighbors.append(factor_node)

class FactorNode:
    def __init__(self, name, function, factor_type='smoothing'):
        self.name = name
        self.function = function
        self.factor_type = factor_type
        self.neighbors = []

    def add_neighbor(self, variable_node):
        self.neighbors.append(variable_node)

class FactorGraph:
    def __init__(self):
        self.variables = []
        self.factors = []
        self.graph = nx.Graph()
        self.num_loops = 0
        self.num_priors = 0
        self.is_tree = False
        self.is_grid = False
        self.closest_prior = None

    def add_variable(self, variable_name, belief_discretisation):
        variable_node = VariableNode(variable_name, belief_discretisation)
        self.variables.append(variable_node)
        self.graph.add_node(variable_name, node_type='variable')# , belief=belief)
        return variable_node

    def add_factor(self, variable_list, function, factor_type='smoothing'):
        # Check for duplicate factors
        if self.factor_exists(variable_list):
            return None
        # Create a unique factor name based on the variable names
        factor_name = 'f'+'_'.join(variable.name for variable in variable_list)
        factor_node = FactorNode(factor_name, function, factor_type)
        self.factors.append(factor_node)
        self.graph.add_node(factor_name, function=function, node_type='factor', factor_type=factor_type)
        for variable in variable_list: self.connect(factor_node, variable)
        return factor_node

    def connect(self, variable, factor):
        # Connect variable to factor in both directions
        self.graph.add_edge(variable.name, factor.name)
        variable.add_neighbor(factor)
        factor.add_neighbor(variable)

    def factor_exists(self, variable_list):
        variable_names = set(variable.name for variable in variable_list)
        for factor in self.factors:
            if set(n.name for n in factor.neighbors) == variable_names:
                return True
        return False


''' functions '''
#TODO: you shouldn't need belief_discretisation here
def add_variables_to_graph(graph, num_variables, belief_discretisation):
    # Add variable nodes
    for i in range(num_variables):
        graph.add_variable(f'X{i + 1}', belief_discretisation)

def add_priors_to_graph(graph, num_priors, measurement_range, prior_location):
    # Add prior factors
    # if it's a tree and you want priors on the leaf nodes
    if (graph.is_tree and prior_location == 'leaf'):
        for variable in graph.variables:
            if len(variable.neighbors) == 1:
                random_prior_function = dm.create_random_prior_distribution(measurement_range)
                graph.add_factor([variable], random_prior_function, factor_type='prior')
    # Otherwise if it's loopy or a tree with a root prior
    elif prior_location == 'root' or prior_location == 'random':
        # add a random prior to all leaf nodes
        for i in range(num_priors):
            random_prior_function = dm.create_random_prior_distribution(measurement_range)
            graph.add_factor([graph.variables[i*int((len(graph.variables)/num_priors))]], random_prior_function, factor_type='prior')
            graph.num_priors += 1
    elif prior_location == 'top':
        i = 0
        for var in graph.variables:
            if i < num_priors:
                random_prior_function = dm.create_random_prior_distribution(measurement_range)
                graph.add_factor([var], random_prior_function, factor_type='prior')
                graph.num_priors += 1
                i+=1
            else:
                break
    elif prior_location == 'edges':
        # add priors to the top left corner
        var_cols = int(np.ceil(np.sqrt(len(graph.variables))))
        top_priors = num_priors//2 + num_priors%2
        top_prior_cols = int(np.ceil(np.sqrt(top_priors)))
        for i in range(top_priors):
            prior_function = dm.create_random_prior_distribution(measurement_range)
            graph.add_factor([graph.variables[i+i//top_prior_cols*(var_cols-top_prior_cols)]], prior_function, factor_type='prior')

        # add priors to the bottom right corner
        bottom_priors = num_priors//2
        bottom_prior_cols = int(np.ceil(np.sqrt(bottom_priors)))
        for i in range(bottom_priors):
            prior_function = dm.create_random_prior_distribution(measurement_range)
            graph.add_factor([graph.variables[-1-i-(i//bottom_prior_cols)*(var_cols-bottom_prior_cols)]], prior_function, factor_type='prior')



def add_loopy_pairwise_factors(graph, num_loops, measurement_range):
    num_variables = len(graph.variables)
    belief_discretisation = len(graph.variables[0].belief)
    pairwise_function = dm.create_smoothing_factor_distribution(belief_discretisation)
    for i in range(num_variables):
        # add factors between adjacent variables
        pairwise_function = dm.create_smoothing_factor_distribution(belief_discretisation)
        
        # if it's a chain, connect all the variables in a line
        if i < num_variables - 1:
            graph.add_factor([graph.variables[i], graph.variables[i + 1]], function=pairwise_function)

        # if it's loopy, connect the last variable to the first
        if num_loops > 0: 
            if i == num_variables - 1:
                graph.add_factor([graph.variables[i], graph.variables[0]],function=pairwise_function)
                graph.num_loops+=1

            # if there's more loops than 1, evenly space them around the outer loop
    for i in range(1, num_loops):
        graph.add_factor([graph.variables[int(i*num_variables/(2*num_loops))],
                        graph.variables[int(num_variables-((i*num_variables)/(2*num_loops)))]],
                        function=pairwise_function
                        )
        graph.num_loops += 1

def add_tree_pairwise_factors(graph, branching_factor, branching_probability):
    variables = graph.variables
    num_variables = len(variables)
    belief_discretisation = len(variables[0].belief)
    queue = [variables[0]]
    next_var_idx = 1

    while queue and next_var_idx < num_variables:
        layer_size = len(queue)
        for i in range(layer_size):
            parent = queue.pop(0)
            # Decide if this parent should branch
            should_branch = (dm.rng.random() < branching_probability) or (i == layer_size - 1 and next_var_idx < num_variables)
            num_children = branching_factor if should_branch else 0
            # If this is the last parent in the layer and there are still variables left, force at least one child
            if i == layer_size - 1 and next_var_idx < num_variables and num_children == 0:
                num_children = 1
            for _ in range(num_children):
                if next_var_idx >= num_variables:
                    break
                child = variables[next_var_idx]
                next_var_idx += 1
                pairwise_function = dm.create_smoothing_factor_distribution(
                    belief_discretisation, prior=dm.create_random_prior_distribution(child.belief)
                )
                graph.add_factor([parent, child], function=pairwise_function)
                queue.append(child)

def add_grid_pairwise_factors(graph):
    num_variables = len(graph.variables)
    grid_cols = int(np.ceil(np.sqrt(num_variables)))  # Assuming a square grid for simplicity
    belief_discretisation = len(graph.variables[0].belief)
    i=0
    for i in range(num_variables):
        current_var = graph.variables[i]
        below_var = None
        right_var = None
        # determine variable to the right of the current one, if it exists
        if i+1 < num_variables: 
            right_var = graph.variables[i+1]
        # determine variable below the current one, if it exists
        if i+grid_cols < num_variables: 
            below_var = graph.variables[i+grid_cols]
        # create a pairwise factor function for the right and below variables
        pairwise_function_1 = dm.create_smoothing_factor_distribution(
            belief_discretisation
        )
        pairwise_function_2 = dm.create_smoothing_factor_distribution(
            belief_discretisation
        )
        
        # Connect to variable below the current one (if it's correct).
        if below_var:
            graph.add_factor([current_var, below_var], pairwise_function_1)
        
        # Connect to variables to the right of the current one (if it's correct). 
        if right_var and (i+1)%grid_cols!=0: 
            graph.add_factor([current_var, right_var], pairwise_function_2)
            
    

def add_pairwise_factors_to_graph(graph, num_loops, measurement_range, branching_factor, branching_probability):
    if graph.is_tree:            
        add_tree_pairwise_factors(graph, branching_factor, branching_probability)
    elif graph.is_grid:
        add_grid_pairwise_factors(graph)
    else:
        add_loopy_pairwise_factors(graph, num_loops, measurement_range)


#TODO: make the number of arguments being passed here more efficient
def build_factor_graph(num_variables, num_priors, num_loops, graph_type, measurement_range, branching_factor, branching_probability, prior_location):
    # Create a factor graph
    graph = FactorGraph()
    belief_discretisation = len(measurement_range)
    if graph_type == 'Tree': graph.is_tree = True
    elif graph_type == 'Grid': graph.is_grid =  True

    add_variables_to_graph(graph, num_variables, belief_discretisation)
    add_pairwise_factors_to_graph(graph, num_loops, measurement_range, branching_factor, branching_probability)
    add_priors_to_graph(graph, num_priors, measurement_range, prior_location)

    return graph