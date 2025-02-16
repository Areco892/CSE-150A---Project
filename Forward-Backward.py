import numpy as np

class HMM:
    # Constructor
    def __init__(self, trans_matrix, emission_matrix, initial_state = None):
        self.trans_matrix = trans_matrix
        self.emission_matrix = emission_matrix
        self.initial_state = initial_state or np.array([0.5, 0.5])
    
    # 
    def em_dist(self, evidence):
        if evidence:
            return self.emission_matrix.T[0]
        else:
            return self.emission_matrix.T[1]
        
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

umbrella_trasition_M = np.array([[0.7, 0.3], [0.3, 0.7]])
umbrella_emission_M = np.array([[0.9, 0.1], [0.2, 0.8]])
umbrellaHMM = HMM(umbrella_trasition_M, umbrella_emission_M)

evidence = [True, True, False, True, True]

print("Forward Algorithm")
forward_inference = np.array([0.5, 0.5])
print(f"Initial Forward Inference: {forward_inference}")
forward_solution = list()
for ev in evidence[::-1]:
  forward_inference = forwardStep(umbrellaHMM, forward_inference, ev)
  print(forward_inference)
  forward_solution.append(forward_inference)
print(f"Most probable forward path: {np.argmax(forward_solution, axis = 1)}")

print("\nBackward Algorithm")
backward_inference = np.array([1.0, 1.0])
print(f"Initial Backward Inference: {backward_inference}")
backward_solution = list()
for ev in evidence[::-1]:
  backward_inference = backwardStep(umbrellaHMM, backward_inference, ev)
  print(backward_inference)
  backward_solution.append(backward_inference)
print(f"Most probable backward path: {np.argmax(backward_solution, axis = 1)}")

print("\nForward-Backward Algorithm")
solution = smoothingStep(forward_solution, backward_solution)
for sol in solution:
    print(sol)
print(f"Most probable path: {np.argmax(solution, axis = 1)}")