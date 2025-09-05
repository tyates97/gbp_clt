
# # External  libraries
# import numpy as np

# # Local modules
# import distribution_management as dm


# def run_belief_propagation(graph, num_iterations, bp_pass_direction='Forward pass'):
#     '''
#     runs the belief propagation algorithm on a given factor graph for a given number of iterations
#     '''

#     ### STEP 0: Initialise all messages & beliefs (start with uniform messages)
#     print("Running BP: Initialising messages...")
#     messages = {}
#     discretisation = len(graph.variables[0].belief)

#     for variable in graph.variables:
#         for factor in variable.neighbors:           # test if we can initialise the graph much quicker.
#         # for factor in graph.factors:
#             messages[(factor, variable)] = dm.normalise(np.ones(discretisation))
#             messages[(variable, factor)] = dm.normalise(np.ones(discretisation))

#     # Helper: send variable-to-factor message
#     def send_var_to_factor(variable, exclude_factor):
#         msg = np.ones(discretisation)
#         for factor in variable.neighbors:
#             if factor != exclude_factor:
#                 msg *= messages[(factor, variable)]
#         return dm.normalise(msg)

#     # Helper: send factor-to-variable message (pairwise factors)
#     def send_factor_to_var(factor, exclude_variable):
#         idx = factor.neighbors.index(exclude_variable)
#         other_variable = factor.neighbors[1 - idx]
#         incoming = messages[(other_variable, factor)]
#         msg = np.zeros(discretisation)
#         if idx == 0:
#             for i in range(discretisation):
#                 msg[i] = np.sum(factor.function[i, :] * incoming)
#         else:
#             for i in range(discretisation):
#                 msg[i] = np.sum(factor.function[:, i] * incoming)
#         return dm.normalise(msg)

#     # Forward pass: from root to leaves
#     def forward(variable, parent_factor=None, visited=None):
#         if visited is None:
#             visited = set()
#         visited.add(variable)
#         for factor in variable.neighbors:
#             if factor == parent_factor:
#                 continue
#             other_variables = [v for v in factor.neighbors if v != variable]
#             if not other_variables:
#                 continue
#             other_variable = other_variables[0]  # Assuming pairwise factors for now
#             # Update messages
#             messages[(variable, factor)] = send_var_to_factor(variable, factor)
#             messages[(factor, other_variable)] = send_factor_to_var(factor, other_variable)
#             if other_variable not in visited:
#                 forward(other_variable, factor, visited)

#     # Backward pass: from leaves to root
#     def backward(variable, parent_factor=None, visited=None):
#         if visited is None:
#             visited = set()
#         visited.add(variable)
#         # For each neighbor factor except the parent
#         for factor in variable.neighbors:
#             if factor == parent_factor:
#                 continue
#             # Find the other variable connected to this factor
#             other_variables = [v for v in factor.neighbors if v != variable]
#             if not other_variables:
#                 continue
#             other_variable = other_variables[0]  # Assuming pairwise factors for now
#             if other_variable not in visited:
#                 backward(other_variable, factor, visited)
#             # After all children have sent messages, update messages up to parent
#             messages[(other_variable, factor)] = send_var_to_factor(other_variable, factor)
#             messages[(factor, variable)] = send_factor_to_var(factor, variable)


#     # Initialise messages from all prior factors
#     for factor in graph.factors:
#         if factor.factor_type == 'prior':
#             # Assuming prior has one neighbor variable
#             messages[(factor, factor.neighbors[0])] = factor.function

#     ### STEP 0.1 - TREE GRAPHS: If the graph is a tree or a chain, run forward and backward passes
#     print("Running BP: Iterating loopy belief propagation...")
#     if graph.is_tree or (graph.num_loops == 0 and graph.num_priors == 1):
#         root = graph.variables[0]
        
#         if bp_pass_direction == 'Forward pass':
#             forward (root)
#         elif bp_pass_direction == 'Backward pass':
#             backward(root)
#         elif bp_pass_direction == 'Both':
#             backward(root)
#             forward(root)

#         # Update beliefs
#         for variable in graph.variables:
#             belief = np.ones(discretisation)
#             for factor in variable.neighbors:
#                 belief *= messages[(factor, variable)]
#             variable.belief = dm.normalise(belief)
#         return graph


#     ### STEP 0.2 - LOOPY GRAPHS: For loopy, non-chain graphs, run normal, iterative belief propagation
#     else:
#         # Begin iterating
#         for iteration in range(num_iterations):

#             ### STEP 1: Update messages from variables to factors
#             for variable in graph.variables:
#                 for factor in variable.neighbors:
#                     # Multiply all incoming messages except from the current factor
#                     incoming_msgs = np.ones(discretisation)
#                     for other_factors in variable.neighbors:
#                         if other_factors != factor:
#                             incoming_msgs *= messages[(other_factors, variable)]

#                     # Normalize and send message to the factor
#                     messages[(variable, factor)] = dm.normalise(incoming_msgs)

#             ### STEP 2: Update messages from factors to variables
#             for factor in graph.factors:
#                 # if this is a prior factor, we can pass the factor's function directly to the variable
#                 if factor.factor_type == 'prior':
#                     messages[(factor, factor.neighbors[0])] = factor.function
#                     continue

#                 #TODO: if you wanted more complex topology, you could edit the code block below to incorporate n-neighbours
#                 for neighbor in factor.neighbors:
#                     other_variable = None
#                     for v_in_factor in factor.neighbors:
#                         if v_in_factor != neighbor:
#                             other_variable = v_in_factor
#                             break

#                     incoming_from_other_var = messages[(other_variable, factor)]
#                     msg_to_var = np.zeros(discretisation)

#                     if neighbor == factor.neighbors[0]:
#                         # Message to the 'row' variable of factor.function
#                         # msg[s_row] = sum_{s_col} factor.function[s_row, s_col] * incoming_from_other_var[s_col]
#                         for id_state_target_var in range(discretisation):
#                             msg_to_var[id_state_target_var] = np.sum(factor.function[id_state_target_var, :] * incoming_from_other_var)
#                     else: # neighbor == factor.neighbors[1]
#                         # Message to the 'column' variable of factor.function
#                         # msg[s_col] = sum_{s_row} factor.function[s_row, s_col] * incoming_from_other_var[s_row]
#                         for id_state_target_var in range(discretisation):
#                              msg_to_var[id_state_target_var] = np.sum(factor.function[:, id_state_target_var] * incoming_from_other_var)

#                     messages[(factor, neighbor)] = dm.normalise(msg_to_var)

#             print(f"Running BP: iteration {iteration+1}/{num_iterations} complete.")

#             # STEP 3: Update beliefs for all variables
#             for variable in graph.variables:
#                 # Beliefs are proportional to the product of all incoming messages
#                 belief = np.ones(discretisation)
#                 for neighbor in variable.neighbors:
#                     belief *= messages[(neighbor, variable)]
#                 variable.belief = dm.normalise(belief)

#     return graph






''' BEGIN TEST '''

# External libraries
import numpy as np
import numba

# Local modules
import distribution_management as dm

# @numba.jit(nopython=True)
def _run_bp_numba(num_iterations, discretisation,
                  factor_to_var_msgs, var_to_factor_msgs, beliefs,
                  factor_connections, var_neighbors, factor_functions, priors, prior_indices,
                  var_neighbor_to_factor_neighbor_idx, factor_to_var_neighbor_idx):
    """
    A Numba-jitted function to run the core belief propagation loop.
    This function uses sparse message arrays (indexed by neighbor, not by node ID)
    to handle large graphs efficiently.
    """
    num_variables = var_neighbors.shape[0]
    num_factors = factor_connections.shape[0]

    # Handle prior factors (which are unary)
    for i in range(prior_indices.shape[0]):
        prior_factor_idx = prior_indices[i]
        # The message from a prior is just the prior itself. It goes to neighbor 0.
        factor_to_var_msgs[prior_factor_idx, 0, :] = priors[i]

    for iteration in range(num_iterations):
        print(f"BP Stage 2: Running iteration {iteration+1}/{num_iterations}...")   
        ### STEP 1: Update messages from variables to factors (using message division)
        for i in range(num_variables):
            # Calculate the product of all incoming messages to this variable
            product_of_incoming = np.ones(discretisation)
            for j in range(var_neighbors.shape[1]):
                factor_idx = var_neighbors[i, j]
                if factor_idx == -1: break
                
                # Find the message from this factor to variable i
                factor_n_idx = var_neighbor_to_factor_neighbor_idx[i, j]
                product_of_incoming *= factor_to_var_msgs[factor_idx, factor_n_idx, :]

            # Calculate outgoing message to each neighbor by dividing the total product
            # by the message that came from that neighbor.
            for j in range(var_neighbors.shape[1]):
                factor_idx = var_neighbors[i, j]
                if factor_idx == -1: break

                factor_n_idx = var_neighbor_to_factor_neighbor_idx[i, j]
                incoming_msg = factor_to_var_msgs[factor_idx, factor_n_idx, :]
                
                # Avoid division by zero
                outgoing_msg = np.ones(discretisation)
                for k in range(discretisation):
                    if incoming_msg[k] > 1e-8:
                        outgoing_msg[k] = product_of_incoming[k] / incoming_msg[k]
                
                s = np.sum(outgoing_msg)
                if s > 0:
                    var_to_factor_msgs[i, j, :] = outgoing_msg / s

        ### STEP 2: Update messages from factors to variables
        for i in range(num_factors):
            # Handle pairwise smoothing factors
            if factor_connections[i, 1] != -1:
                var1_idx = factor_connections[i, 0]
                var2_idx = factor_connections[i, 1]
                
                # Find the incoming messages from each variable
                var1_n_idx = factor_to_var_neighbor_idx[i, 0]
                var2_n_idx = factor_to_var_neighbor_idx[i, 1]
                
                incoming_from_var1 = var_to_factor_msgs[var1_idx, var1_n_idx, :]
                incoming_from_var2 = var_to_factor_msgs[var2_idx, var2_n_idx, :]
                
                # Message to var1 (neighbor 0 of the factor)
                msg_to_var1 = np.dot(factor_functions[i], incoming_from_var2)
                s1 = np.sum(msg_to_var1)
                if s1 > 0:
                    factor_to_var_msgs[i, 0, :] = msg_to_var1 / s1
                
                # Message to var2 (neighbor 1 of the factor)
                msg_to_var2 = np.dot(factor_functions[i].T, incoming_from_var1)
                s2 = np.sum(msg_to_var2)
                if s2 > 0:
                    factor_to_var_msgs[i, 1, :] = msg_to_var2 / s2

    ### STEP 3: Update final beliefs
    for i in range(num_variables):
        belief = np.ones(discretisation)
        for j in range(var_neighbors.shape[1]):
            factor_idx = var_neighbors[i, j]
            if factor_idx == -1: break
            
            factor_n_idx = var_neighbor_to_factor_neighbor_idx[i, j]
            belief *= factor_to_var_msgs[factor_idx, factor_n_idx, :]
        
        s = np.sum(belief)
        if s > 0:
            beliefs[i, :] = belief / s
            
    return beliefs


def run_belief_propagation(graph, num_iterations):
    """
    Wrapper function to run belief propagation.
    It converts the object-oriented graph into NumPy arrays, calls the fast
    Numba-jitted core function, and then updates the graph objects with the results.
    """
    print("BP Stage 1: Converting graph to numerical representation...")

    num_variables = len(graph.variables)
    num_factors = len(graph.factors)
    discretisation = len(graph.variables[0].belief)

    var_map = {var: i for i, var in enumerate(graph.variables)}
    factor_map = {factor: i for i, factor in enumerate(graph.factors)}

    factor_connections = np.zeros((num_factors, 2), dtype=np.int32) - 1
    factor_functions = np.zeros((num_factors, discretisation, discretisation))
    priors, prior_indices = [], []

    max_neighbors = 0
    for var in graph.variables:
        if len(var.neighbors) > max_neighbors:
            max_neighbors = len(var.neighbors)
    var_neighbors = np.zeros((num_variables, max_neighbors), dtype=np.int32) - 1

    for factor, i in factor_map.items():
        if factor.factor_type == 'prior':
            factor_connections[i, 0] = var_map[factor.neighbors[0]]
            priors.append(factor.function)
            prior_indices.append(i)
        else:
            factor_connections[i, 0] = var_map[factor.neighbors[0]]
            factor_connections[i, 1] = var_map[factor.neighbors[1]]
            factor_functions[i, :, :] = factor.function

    var_to_neighbor_map = [{} for _ in range(num_variables)]
    for var, i in var_map.items():
        for j, neighbor_factor in enumerate(var.neighbors):
            f_idx = factor_map[neighbor_factor]
            var_neighbors[i, j] = f_idx
            var_to_neighbor_map[i][f_idx] = j

    priors = np.array(priors)
    prior_indices = np.array(prior_indices, dtype=np.int32)

    # Create lookup tables for sparse indexing
    var_neighbor_to_factor_neighbor_idx = np.zeros_like(var_neighbors)
    for v_idx in range(num_variables):
        for n_idx in range(max_neighbors):
            f_idx = var_neighbors[v_idx, n_idx]
            if f_idx == -1: break
            if factor_connections[f_idx, 0] == v_idx:
                var_neighbor_to_factor_neighbor_idx[v_idx, n_idx] = 0
            elif factor_connections[f_idx, 1] == v_idx:
                var_neighbor_to_factor_neighbor_idx[v_idx, n_idx] = 1

    factor_to_var_neighbor_idx = np.zeros_like(factor_connections)
    for f_idx in range(num_factors):
        v1_idx = factor_connections[f_idx, 0]
        if v1_idx != -1:
            factor_to_var_neighbor_idx[f_idx, 0] = var_to_neighbor_map[v1_idx][f_idx]
        v2_idx = factor_connections[f_idx, 1]
        if v2_idx != -1:
            factor_to_var_neighbor_idx[f_idx, 1] = var_to_neighbor_map[v2_idx][f_idx]

    # **FIX**: Initialize message arrays with sparse shapes to avoid memory error
    factor_to_var_msgs = np.ones((num_factors, 2, discretisation)) / discretisation
    var_to_factor_msgs = np.ones((num_variables, max_neighbors, discretisation)) / discretisation
    beliefs = np.ones((num_variables, discretisation)) / discretisation

    print("BP Stage 2: Running BP iterations...")
    
    final_beliefs = _run_bp_numba(num_iterations, discretisation,
                                  factor_to_var_msgs, var_to_factor_msgs, beliefs,
                                  factor_connections, var_neighbors, factor_functions,
                                  priors, prior_indices,
                                  var_neighbor_to_factor_neighbor_idx, factor_to_var_neighbor_idx)

    print("BP Stage 3: Updating graph objects with final beliefs...")
    for i, variable in enumerate(graph.variables):
        variable.belief = final_beliefs[i, :]

    return graph


''' END TEST '''