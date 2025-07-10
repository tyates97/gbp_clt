"""
factor_graph.py
"""
# External libraries
import networkx as nx
import numpy as np

# Local modules
import distribution_management as dm


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

def add_priors_to_graph(graph, num_priors, prior_function, measurement_range, tree_priors):
    # Add prior factors
    # if it's a tree and you want priors on the leaf nodes
    if (graph.is_tree and tree_priors == 'leaf priors'):
        for variable in graph.variables:
            if len(variable.neighbors) == 1:
                random_prior_function = dm.create_prior_distribution('random', measurement_range)
                graph.add_factor([variable], random_prior_function, factor_type='prior')
    # Otherwise if it's loopy or a tree with a root prior
    elif num_priors > 0:
        # add a random prior to all leaf nodes
        for i in range(num_priors):
            graph.add_factor([graph.variables[i*int((len(graph.variables)/num_priors))]], prior_function, factor_type='prior')
            graph.num_priors += 1

def add_loopy_pairwise_factors(graph, num_loops, identical_smoothing_functions, measurement_range, prior_function):
    num_variables = len(graph.variables)
    belief_discretisation = len(graph.variables[0].belief)
    pairwise_function = dm.create_smoothing_factor_distribution(belief_discretisation, prior_function)
    for i in range(num_variables):
        # add factors between adjacent variables
        if not identical_smoothing_functions:
            pairwise_function = dm.create_smoothing_factor_distribution(belief_discretisation, prior=dm.create_prior_distribution('random', measurement_range))
        
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
            should_branch = (np.random.rand() < branching_probability) or (i == layer_size - 1 and next_var_idx < num_variables)
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
                    belief_discretisation, prior=dm.create_prior_distribution('random', child.belief)
                )
                graph.add_factor([parent, child], function=pairwise_function)
                queue.append(child)

def add_pairwise_factors_to_graph(graph, num_loops, identical_smoothing_functions, measurement_range, prior_function, branching_factor, branching_probability):
    num_variables = len(graph.variables)
    if graph.is_tree:            
        add_tree_pairwise_factors(graph, branching_factor, branching_probability)
    else:
        add_loopy_pairwise_factors(graph, num_loops, identical_smoothing_functions, measurement_range, prior_function)


#TODO: make the number of arguments being passed here more efficient
def build_factor_graph(num_variables, num_priors, num_loops, graph_type, identical_smoothing_functions, measurement_range, prior_distribution_type, branching_factor, branching_probability, tree_prior_location='root prior'):
    # Create a factor graph
    graph = FactorGraph()
    belief_discretisation = len(measurement_range)
    if graph_type == 'Tree Graph': graph.is_tree = True
    prior_function = dm.create_prior_distribution(prior_distribution_type, measurement_range)

    add_variables_to_graph(graph, num_variables, belief_discretisation)
    add_pairwise_factors_to_graph(graph, num_loops, identical_smoothing_functions, measurement_range, prior_function, branching_factor, branching_probability)
    add_priors_to_graph(graph, num_priors, prior_function, measurement_range, tree_prior_location)

    return graph