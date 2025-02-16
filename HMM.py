import numpy as np


class HMM:
    # Constructor
    def __init__(self, trans_matrix, emission_matrix, initial_state = np.array([0.5, 0.5])):
        self.trans_matrix = trans_matrix
        self.emission_matrix = emission_matrix
        self.initial_state = initial_state
    
    # 
    def em_dist(self, evidence):
        if evidence == 0:
            return self.emission_matrix.T[0]
        elif evidence == 1:
            return self.emission_matrix.T[1]
        else:
            return self.emission_matrix.T[2]
        
# Implement forward algorithm
def forwardStep(myHMM, forward_inference, ev):
    # Set-up step: Get all the terms needed for next forward inference
    observation_matrix = np.multiply(myHMM.em_dist(ev), np.array([[1,0],[0,1]]))
    transition_matrix = myHMM.trans_matrix.T # must transpose for forward inference
    forward_inference_matrix = forward_inference
    
    # Matrix multiplication step: multiply all the matrices involved
    prediction = observation_matrix @ transition_matrix @ forward_inference_matrix
    
    # Normalizing step: Normalize to bring the sum of the probabilities to 1
    normalizing_constant = prediction[0] + prediction[1]
    normalized_prediction = np.divide(prediction, normalizing_constant)
    return  normalized_prediction

# Implement backward algorithm
def backwardStep(myHMM, backward_inference, ev):
    # Set-up step: Get all the terms needed for next backward inference
    observation_matrix = np.multiply(myHMM.em_dist(ev), np.array([[1,0],[0,1]]))
    transition_matrix = myHMM.trans_matrix
    backward_inference_matrix = backward_inference

    # Matrix multiplication step: multiply all the matrices involved
    prediction = transition_matrix @ observation_matrix @ backward_inference_matrix

    # Normalizing step: Normalize to bring the sum of the probabilities to 1
    normalizing_constant = prediction[0] + prediction[1]
    normalized_prediction = np.divide(prediction, normalizing_constant)
    return  normalized_prediction

# Implement forward-backward algorithm
def smoothingStep(forward_solution, backward_solution):
    # Set-up step: Get all the terms needed for next backward inference
    solution = list()
    entries = len(forward_solution)
    for i in range(entries):
        forward = forward_solution[i]
        backward = backward_solution[entries - i - 1]
        prediction = np.multiply(forward, backward)
        normalized_prediction = np.divide(prediction, (prediction[0] + prediction[1]))
        solution.append(normalized_prediction)
    return solution

# Implement viterbi algorithm
def viterbi(myHMM, evidence):
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

# Creating the HMM model
traffic_trans_matrix = np.array([[0.3125    , 0.59210526, 0.09539474], 
                                 [0.10851319, 0.74340528, 0.14808153], 
                                 [0.02691924, 0.24825523, 0.72482552]])

traffic_emission_matrix = np.array([[0.19471698,  0.80528302, 0.        ], 
                                    [0.03425168,  0.6820551 , 0.28369322], 
                                    [0.        ,  0.02272727, 0.97727273]])

traffic_initial_state = np.array([1/3, 1/3, 1/3])
trafficHMM = HMM(traffic_trans_matrix, traffic_emission_matrix, traffic_initial_state)

# List the evidences
evidence = [0, 1, 2, 1, 2]

# Get the most probable sequence of hidden states
print(viterbi(trafficHMM, evidence))