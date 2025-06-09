
# External  libraries
import numpy as np

# Local modules
import distribution_management as dm


def run_belief_propagation(graph, num_iterations):

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

    # Initialise the message from prior to first variable - this should just be the prior
    if graph.factors and graph.variables and graph.factors[0].factor_type == 'prior':
        messages[(graph.factors[0], graph.factors[0].neighbors[0])] = graph.factors[0].function

    ### STEP 0.1 - CHAIN GRAPHS: For chain graphs, run a forward/backward pass algorithm for faster convergence
    if graph.num_loops == 0 and graph.num_priors == 1:
        ### --- CHAIN: FORWARD PASS (FP) ---
        # Iterate through the chain to calculate the messages:
        for variable in graph.variables[:-1]:

            if not variable.neighbors:      # skip if variable has no connections
                continue

            factor_function = variable.neighbors[1].function
            incoming_f2v_message = messages[(variable.neighbors[0], variable)]

            # Then find message from factor_node to v2
            outgoing_f2v_message = factor_function.T @ incoming_f2v_message
            messages[(variable.neighbors[1], variable.neighbors[1].neighbors[1])] = dm.normalise(outgoing_f2v_message)


        ### --- CHAIN: BACKWARD PASS (BP) ---
        # iterate through the chain to calculate the messages
        for variable in reversed(graph.variables[1:]):
            factor_function = variable.neighbors[0].function
            # Message from v2 to factor_node
            # if last variable, this is np.ones(), otherwise use the previous message
            if len(variable.neighbors) == 1:
                incoming_f2v_message = dm.normalise(np.ones(discretisation))
            else:
                incoming_f2v_message = messages[(variable.neighbors[1], variable)]

            # Now compute message from factor_node to v1
            outgoing_f2v_message = factor_function.T @ incoming_f2v_message
            messages[(variable.neighbors[0], variable.neighbors[0].neighbors[0])] = dm.normalise(outgoing_f2v_message)


        ### --- CHAIN: FINAL BELIEF UPDATE ---
        # After forward and backward passes, update beliefs for all variables
        for variable in graph.variables:
            belief = np.ones(discretisation)

            for factor in variable.neighbors:
                # We need the message from the factor to the variable
                belief *= messages[(factor, variable)]
            belief = dm.normalise(belief)
            variable.belief = belief


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