"""
factor_graph.py
"""
# External libraries
import networkx as nx
import numpy as np
import numba

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
        self.grid_cols = None
        self.closest_prior = None

    def add_variable(self, variable_name, belief_discretisation):
        variable_node = VariableNode(variable_name, belief_discretisation)
        self.variables.append(variable_node)
        self.graph.add_node(variable_name, node_type='variable')# , belief=belief)
        return variable_node

    def add_factor(self, variable_list, function, factor_type='smoothing'):
        # # Check for duplicate factors # now redundant with calculate grid connections function
        # if self.factor_exists(variable_list):
        #     return None
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
    
        #TODO: you shouldn't need belief_discretisation here
    def add_variables(self, num_variables, belief_discretisation):
        print("Adding variables to graph...")
        # Add variable nodes
        for i in range(num_variables):
            self.add_variable(f'X{i + 1}', belief_discretisation)

    def add_leaf_priors(self, measurement_range):
        for variable in self.variables:
            if len(variable.neighbors) == 1:
                random_prior_function = dm.create_random_prior_distribution(measurement_range)
                self.add_factor([variable], random_prior_function, factor_type='prior')

    def add_priors(self, num_priors, measurement_range, prior_location):
        print("Adding prior factors to graph...")
        # if it's a tree and you want priors on the leaf nodes
        if (self.is_tree and prior_location == 'leaf'):
            self.add_leaf_priors(measurement_range)

        # if it's a tree and you want a prior on the root node
        elif isinstance(prior_location, str) and (prior_location == 'root' or prior_location == 'random'):
            # add a random prior to all leaf nodes
            for i in range(num_priors):
                random_prior_function = dm.create_random_prior_distribution(measurement_range)
                self.add_factor([self.variables[i*int((len(self.variables)/num_priors))]], random_prior_function, factor_type='prior')
                self.num_priors += 1
        
        elif isinstance(prior_location, str) and prior_location == 'top':
            i = 0
            for var in self.variables:
                if i < num_priors:
                    random_prior_function = dm.create_random_prior_distribution(measurement_range)
                    self.add_factor([var], random_prior_function, factor_type='prior')
                    self.num_priors += 1
                    i+=1
                else:
                    break
    
        elif isinstance(prior_location, str) and prior_location == 'corners':
            # add priors to the top left corner
            var_cols = int(np.ceil(np.sqrt(len(self.variables))))
            top_priors = num_priors//2 + num_priors%2
            top_prior_cols = int(np.ceil(np.sqrt(top_priors)))
            for i in range(top_priors):
                prior_function = dm.create_random_prior_distribution(measurement_range)
                self.add_factor([self.variables[i+i//top_prior_cols*(var_cols-top_prior_cols)]], prior_function, factor_type='prior')

            # add priors to the bottom right corner
            bottom_priors = num_priors//2
            bottom_prior_cols = int(np.ceil(np.sqrt(bottom_priors)))
            for i in range(bottom_priors):
                prior_function = dm.create_random_prior_distribution(measurement_range)
                self.add_factor([self.variables[-1-i-(i//bottom_prior_cols)*(var_cols-bottom_prior_cols)]], prior_function, factor_type='prior')
        
        elif isinstance(prior_location, np.ndarray):
            flat_prior_locations = prior_location.flatten() # Flatten the 2D array to 1D
            flat_depth_map = cfg.depth_map_meters.flatten()
            for i, variable in enumerate(self.variables):
                if flat_prior_locations[i]:
                    prior_function = dm.create_random_prior_distribution(cfg.measurement_range, mean=flat_depth_map[i], prior_width=32)
                    self.add_factor([variable], prior_function, factor_type='prior')
                    self.num_priors += 1

    def add_priors_from_pdf(self, pdf_volume):
        print("Adding prior factors...")
        height, width, _ = pdf_volume.shape
        flat_pdf = pdf_volume.reshape(-1, pdf_volume.shape[2])

        for i, variable in enumerate(self.variables):
            prior_function = flat_pdf[i]
            self.add_factor([variable], prior_function, factor_type="prior")
            self.num_priors += 1

        # for y_pos in pdf_volume.shape[0]:
        #     for x_pos in pdf_volume.shape[1]:
        #         graph_idx = (y_pos-1)*num_cols+x_pos
        #         prior_function = pdf_volume[y_pos][x_pos][:]
        #         self.add_factor([self.variables[graph_idx]], prior_function, factor_type='prior')

        return self


    def add_loopy_pairwise_factors(self, num_loops, measurement_range):
        num_variables = len(self.variables)
        belief_discretisation = len(self.variables[0].belief)
        pairwise_function = dm.create_smoothing_factor_distribution(belief_discretisation)
        for i in range(num_variables):
            # add factors between adjacent variables
            pairwise_function = dm.create_smoothing_factor_distribution(belief_discretisation)
            
            # if it's a chain, connect all the variables in a line
            if i < num_variables - 1:
                self.add_factor([self.variables[i], self.variables[i + 1]], function=pairwise_function)

            # if it's loopy, connect the last variable to the first
            if num_loops > 0: 
                if i == num_variables - 1:
                    self.add_factor([self.variables[i], self.variables[0]],function=pairwise_function)
                    self.num_loops+=1

                # if there's more loops than 1, evenly space them around the outer loop
        for i in range(1, num_loops):
            self.add_factor([self.variables[int(i*num_variables/(2*num_loops))],
                            self.variables[int(num_variables-((i*num_variables)/(2*num_loops)))]],
                            function=pairwise_function
                            )
            self.num_loops += 1

    def add_tree_pairwise_factors(self, branching_factor, branching_probability):
        variables = self.variables
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
                    self.add_factor([parent, child], function=pairwise_function)
                    queue.append(child)

    @staticmethod               # to prevent needing to pass "self" to the function, which it doesn't need
    @numba.jit(nopython=True)   # to speed up the compiler
    def _calculate_grid_connections(num_variables, grid_cols):
        """
        Calculates pairwise connections for a grid graph.
        This function is JIT-compiled with Numba for high performance.
        It only uses primitive types and NumPy arrays.
        """
        # Pre-allocate a NumPy array to store connections. This is much faster
        # than appending to a Python list inside a Numba loop.
        # The maximum number of connections is roughly 2 * num_variables.
        max_connections = 2 * num_variables
        connections_array = np.empty((max_connections, 2), dtype=np.int64)
        count = 0
        print("Calculating pairwise grid connections...")

        for i in range(num_variables):
            # Connection to the right neighbor
            # Check we are not in the last column and not at the last variable
            if (i + 1) % grid_cols != 0 and (i + 1) < num_variables:
                connections_array[count, 0] = i
                connections_array[count, 1] = i + 1
                count += 1
            
            # Connection to the neighbor below
            if i + grid_cols < num_variables:
                connections_array[count, 0] = i
                connections_array[count, 1] = i + grid_cols
                count += 1
                
        # Return a slice of the array containing only the actual connections
        return connections_array[:count]


    def add_grid_pairwise_factors(self, num_cols=None, hist=None):
        """
        Adds pairwise factors to a grid graph. This function orchestrates the process:
        1. Calls a fast, JIT-compiled helper to calculate connection indices.
        2. Loops through the connections in plain Python to create factor objects.
        """
        num_variables = len(self.variables)
        
        if num_cols is None:
            grid_cols = int(np.ceil(np.sqrt(num_variables)))
        else:
            grid_cols = num_cols # Use the provided number of columns
            
        self.grid_cols = grid_cols

        belief_discretisation = len(self.variables[0].belief)

        # 1. Call the fast helper function to get the numerical connection plan
        connections = self._calculate_grid_connections(num_variables, grid_cols)

        print("Creating pairwise factors...")
        # 2. Iterate through the connection plan and create the actual factor objects
        for i, j in connections:
            var1 = self.variables[i]
            var2 = self.variables[j]
            
            # Create a new smoothing function for each factor
            pairwise_function = dm.create_smoothing_factor_distribution(belief_discretisation, hist=hist)
            
            self.add_factor([var1, var2], pairwise_function)

            
            if len(self.factors) % 34000 == 0:
                percentage_processed = int((len(self.factors)/len(connections))*100)    
                print(f"{percentage_processed}% of pairwise factors added.")

        print("All pairwise factors added.")
                
        

    def add_pairwise_factors(self, num_loops, measurement_range, branching_factor, branching_probability, num_cols=None, hist=None):
        if self.is_tree:            
            self.add_tree_pairwise_factors(branching_factor, branching_probability)
        elif self.is_grid:
            self.add_grid_pairwise_factors(num_cols=num_cols, hist=hist)
        else:
            self.add_loopy_pairwise_factors(num_loops, measurement_range)










''' functions '''

#TODO: make the number of arguments being passed here more efficient
def build_factor_graph(num_variables, num_priors, num_loops, graph_type, measurement_range, prior_location, branching_factor=2, branching_probability=1.0, hist=None):
    print("Building factor graph...")
    # Create a factor graph
    graph = FactorGraph()
    belief_discretisation = len(measurement_range)
    if graph_type == 'Tree': graph.is_tree = True
    elif graph_type == 'Grid': graph.is_grid =  True
    graph.add_variables(num_variables, belief_discretisation)
    graph.add_pairwise_factors(num_loops, measurement_range, branching_factor, branching_probability, hist=hist)
    graph.add_priors(num_priors, measurement_range, prior_location)
    return graph

def get_graph_from_pdf_hist(pdf_volume, hist=None):
    print("Building factor graph from PDF volume...")
    height = pdf_volume.shape[0]
    width = pdf_volume.shape[1]
    
    graph = FactorGraph()
    num_variables = height*width
    graph.is_grid = True
    graph.grid_cols = width
    graph.add_variables(num_variables, belief_discretisation=pdf_volume.shape[2])
    graph.add_pairwise_factors(0, cfg.measurement_range, 1, 1, num_cols=graph.grid_cols, hist=hist)
    graph.add_priors_from_pdf(pdf_volume)

    return graph

def save_beliefs(graph):
    """ Save current beliefs from all variables in the graph """
    return {i: var.belief.copy() for i,var in enumerate(graph.variables)}

def restore_beliefs(graph, saved_beliefs):
    """ Restore beliefs to all variables in the graph """
    for i, variable in enumerate(graph.variables):
        variable.belief = saved_beliefs[i].copy()

def save_factor_functions(graph):
    return {i: f.function.copy() for i, f in enumerate(graph.factors) if hasattr(f, "function")}

def restore_factor_functions(graph, saved_functions):
    for i, f in enumerate(graph.factors):
        if i in saved_functions:
            f.function = saved_functions[i].copy()