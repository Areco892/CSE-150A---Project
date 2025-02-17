import numpy as np

"""
Training our Model
"""

class HMM:
    # Constructor
    def __init__(self, trans_matrix, emission_matrix, initial_state = np.array([0.5, 0.5])):
        self.trans_matrix = trans_matrix
        self.emission_matrix = emission_matrix
        self.initial_state = initial_state
    
    # Get the corresponding observation from the emission matrix
    def em_dist(self, evidence):
        if evidence == 0:
            return self.emission_matrix.T[0]
        elif evidence == 1:
            return self.emission_matrix.T[1]
        else:
            return self.emission_matrix.T[2]

    # Implement viterbi algorithm
    def viterbi(self, myHMM, evidence):
        # Gather all the needed matrices to get the conditional probabilities
        initial_state = myHMM.initial_state
        transition_matrix = myHMM.trans_matrix
        emission_matrix = myHMM.emission_matrix
        
        # Initialize the probability matrix to keep track of the probabilities up to the current state
        n_evidence = len(evidence)
        n_states = len(emission_matrix[0])
        prob_matrix = np.zeros((n_states, n_evidence))
        
        # Initialize the initial state and first observation
        for state in range(n_states):
            prob_matrix[state][0] = initial_state[state] * myHMM.em_dist(evidence[0])[state]

        # Iterate through the possible states keeping only the maximum probability for every state
        max = 0
        for ev in range(1, n_evidence):
            for next_state in range(n_states):
                for curr_state in range(n_states):
                    top = prob_matrix[curr_state][ev-1] * transition_matrix[curr_state][next_state] * myHMM.em_dist(evidence[ev])[next_state]
                    if top > max:
                        max = top
                prob_matrix[next_state][ev] = max
                max = 0

        # Generate the path that is most likely to occur given the set of evidences
        path = np.argmax(prob_matrix, axis = 0)

        return path
