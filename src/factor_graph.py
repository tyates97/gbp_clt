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
def build_factor_graph(num_variables, num_priors, num_loops, is_tree, identical_smoothing_functions, measurement_range, prior_distribution_type, gauss_sigma, branching_factor=2):
    # Create a factor graph
    graph = FactorGraph()
    graph.is_tree = is_tree
    belief_discretisation = len(measurement_range)
    prior_function = dm.create_prior_distribution(prior_distribution_type, measurement_range, gauss_sigma)

    # Add variable nodes
    for i in range(num_variables):
        graph.add_variable(f'X{i + 1}', belief_discretisation)

    # Add prior factors, evenly spaced around the number of variables
    for i in range(num_priors):
        graph.add_factor([graph.variables[i*int((num_variables/num_priors))]], prior_function, factor_type='prior')
        graph.num_priors += 1

    # Add pairwise factors for each connected variable
    pairwise_function = dm.create_smoothing_factor_distribution(belief_discretisation, prior=graph.factors[0].function)
    for i in range(len(graph.variables)):
        # add factors between adjacent variables
        if not identical_smoothing_functions:
            pairwise_function = dm.create_smoothing_factor_distribution(belief_discretisation, prior=dm.create_prior_distribution('random', measurement_range, gauss_sigma))
        
        # if it's a chain, connect all the variables in a line
        if not is_tree and i < len(graph.variables) - 1:
            graph.add_factor([graph.variables[i], graph.variables[i + 1]],
                                  function=pairwise_function
                              )

        # if it's loopy, connect the last variable to the first
        elif not is_tree and num_loops > 0 and i == len(graph.variables) - 1:
            graph.add_factor([graph.variables[i], graph.variables[0]],
                                  function=pairwise_function
                              )
            graph.num_loops+=1

            # if there's more loops than 1, evenly space them around the outer loop
            for i in range(1, num_loops):
                graph.add_factor([graph.variables[int(i*num_variables/(2*num_loops))],
                                 graph.variables[int(num_variables-((i*num_variables)/(2*num_loops)))]],
                                 function=pairwise_function
                                 )
                graph.num_loops += 1

        elif is_tree and i < len(graph.variables) - 1:
            # if it's a tree, connect the variables in a branching structure
            if i % branching_factor == 0:
                # connect to the next branching_factor variables
                for j in range(1, branching_factor + 1):
                    if i + j < len(graph.variables):
                        pairwise_function = dm.create_smoothing_factor_distribution(belief_discretisation, prior=dm.create_prior_distribution('random', measurement_range, gauss_sigma))
                        graph.add_factor([graph.variables[i], graph.variables[i + j]],
                                          function=pairwise_function
                                          )
            # else:
            #     # connect to the previous variable
            #     pairwise_function = dm.create_smoothing_factor_distribution(belief_discretisation, prior=dm.create_prior_distribution('random', measurement_range, gauss_sigma))
            #     graph.add_factor([graph.variables[i], graph.variables[i - 1]],
            #                       function=pairwise_function
            #                       )
    
    return graph