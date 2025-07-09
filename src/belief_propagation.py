
# External  libraries
import numpy as np

# Local modules
import distribution_management as dm


def run_belief_propagation(graph, num_iterations, bp_pass_direction='Forward pass'):
    '''
    runs the belief propagation algorithm on a given factor graph for a given number of iterations
    '''

    ### STEP 0: Initialise all messages & beliefs (start with uniform messages)
    messages = {}
    discretisation = len(graph.variables[0].belief)

    for variable in graph.variables:
        for factor in graph.factors:
            messages[(factor, variable)] = dm.normalise(np.ones(discretisation))
            messages[(variable, factor)] = dm.normalise(np.ones(discretisation))

    # Helper: send variable-to-factor message
    def send_var_to_factor(variable, exclude_factor):
        msg = np.ones(discretisation)
        for factor in variable.neighbors:
            if factor != exclude_factor:
                msg *= messages[(factor, variable)]
        return dm.normalise(msg)

    # Helper: send factor-to-variable message (pairwise factors)
    def send_factor_to_var(factor, exclude_variable):
        idx = factor.neighbors.index(exclude_variable)
        other_variable = factor.neighbors[1 - idx]
        incoming = messages[(other_variable, factor)]
        msg = np.zeros(discretisation)
        if idx == 0:
            for i in range(discretisation):
                msg[i] = np.sum(factor.function[i, :] * incoming)
        else:
            for i in range(discretisation):
                msg[i] = np.sum(factor.function[:, i] * incoming)
        return dm.normalise(msg)

    # Forward pass: from root to leaves
    def forward(variable, parent_factor=None, visited=None):
        if visited is None:
            visited = set()
        visited.add(variable)
        for factor in variable.neighbors:
            if factor == parent_factor:
                continue
            other_variables = [v for v in factor.neighbors if v != variable]
            if not other_variables:
                continue
            other_variable = other_variables[0]  # Assuming pairwise factors for now
            # Update messages
            messages[(variable, factor)] = send_var_to_factor(variable, factor)
            messages[(factor, other_variable)] = send_factor_to_var(factor, other_variable)
            if other_variable not in visited:
                forward(other_variable, factor, visited)

    # Backward pass: from leaves to root
    def backward(variable, parent_factor=None, visited=None):
        if visited is None:
            visited = set()
        visited.add(variable)
        # For each neighbor factor except the parent
        for factor in variable.neighbors:
            if factor == parent_factor:
                continue
            # Find the other variable connected to this factor
            other_variables = [v for v in factor.neighbors if v != variable]
            if not other_variables:
                continue
            other_variable = other_variables[0]  # Assuming pairwise factors for now
            if other_variable not in visited:
                backward(other_variable, factor, visited)
            # After all children have sent messages, update messages up to parent
            messages[(other_variable, factor)] = send_var_to_factor(other_variable, factor)
            messages[(factor, variable)] = send_factor_to_var(factor, variable)


    # Initialise messages from all prior factors
    for factor in graph.factors:
        if factor.factor_type == 'prior':
            # Assuming prior has one neighbor variable
            messages[(factor, factor.neighbors[0])] = factor.function

    ### STEP 0.1 - TREE GRAPHS: If the graph is a tree or a chain, run forward and backward passes
    if graph.is_tree or (graph.num_loops == 0 and graph.num_priors == 1):
        root = graph.variables[0]
        
        if bp_pass_direction == 'Forward pass':
            forward (root)
        if bp_pass_direction == 'Backward pass':
            backward(root)

        # Update beliefs
        for variable in graph.variables:
            belief = np.ones(discretisation)
            for factor in variable.neighbors:
                belief *= messages[(factor, variable)]
            variable.belief = dm.normalise(belief)
        return graph


    ### STEP 0.2 - LOOPY GRAPHS: For loopy, non-chain graphs, run normal, iterative belief propagation
    else:
        # Begin iterating
        for iteration in range(num_iterations):

            ### STEP 1: Update messages from variables to factors
            for variable in graph.variables:
                for factor in variable.neighbors:
                    # Multiply all incoming messages except from the current factor
                    incoming_msgs = np.ones(discretisation)
                    for other_factors in variable.neighbors:
                        if other_factors != factor:
                            incoming_msgs *= messages[(other_factors, variable)]

                    # Normalize and send message to the factor
                    messages[(variable, factor)] = dm.normalise(incoming_msgs)

            ### STEP 2: Update messages from factors to variables
            for factor in graph.factors:
                # if this is a prior factor, we can pass the factor's function directly to the variable
                if factor.factor_type == 'prior':
                    messages[(factor, factor.neighbors[0])] = factor.function
                    continue

                #TODO: if you wanted more complex topology, you could edit the code block below to incorporate n-neighbours
                for neighbor in factor.neighbors:
                    other_variable = None
                    for v_in_factor in factor.neighbors:
                        if v_in_factor != neighbor:
                            other_variable = v_in_factor
                            break

                    incoming_from_other_var = messages[(other_variable, factor)]
                    msg_to_var = np.zeros(discretisation)

                    if neighbor == factor.neighbors[0]:
                        # Message to the 'row' variable of factor.function
                        # msg[s_row] = sum_{s_col} factor.function[s_row, s_col] * incoming_from_other_var[s_col]
                        for id_state_target_var in range(discretisation):
                            msg_to_var[id_state_target_var] = np.sum(factor.function[id_state_target_var, :] * incoming_from_other_var)
                    else: # neighbor == factor.neighbors[1]
                        # Message to the 'column' variable of factor.function
                        # msg[s_col] = sum_{s_row} factor.function[s_row, s_col] * incoming_from_other_var[s_row]
                        for id_state_target_var in range(discretisation):
                             msg_to_var[id_state_target_var] = np.sum(factor.function[:, id_state_target_var] * incoming_from_other_var)

                    messages[(factor, neighbor)] = dm.normalise(msg_to_var)

            # STEP 3: Update beliefs for all variables
            for variable in graph.variables:
                # Beliefs are proportional to the product of all incoming messages
                belief = np.ones(discretisation)
                for neighbor in variable.neighbors:
                    belief *= messages[(neighbor, variable)]
                variable.belief = dm.normalise(belief)

    return graph