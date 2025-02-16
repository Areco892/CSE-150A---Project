import numpy as np
import pandas as pd
import HMM as hmm
import Preprocessing as pr

"""
Evaluating our Model
"""

# Load data
df = pd.read_csv("Traffic.csv")

# Creating the HMM model
traffic_trans_matrix = pr.getTransitionMatrix(df)
traffic_emission_matrix = pr.getEmissionMatrix(df)
traffic_initial_state = np.array([1/3, 1/3, 1/3])
trafficHMM = hmm.HMM(traffic_trans_matrix, traffic_emission_matrix, traffic_initial_state)

# List the observation
observation1 = list()
for i in range(len(df)):
    observation1.append(int(df.iloc[i]["Total"]))
observation = list()
for obs in observation1:
    if obs < 26:
        observation.append(0)
    elif obs < 168:
        observation.append(1)
    else:
        observation.append(2)

# Use Viterbi algorithm and Get the most probable sequence of hidden states
viterbi_path = trafficHMM.viterbi(trafficHMM, observation)
solution = list()
for sol in viterbi_path:
    solution.append(int(sol))

# List of evidences
evidence1 = list()
for i in range(len(df)):
    evidence1.append(df.iloc[i]["Traffic Situation"])
evidence = list()
for ev in evidence1:
    if ev == "low":
        evidence.append(0)
    elif ev == "normal" or ev == "high":
        evidence.append(1)
    else:
        evidence.append(2)

# Compare the Viterbi path to the list of evidences to see how accurate our model is
correct = 0
total = 0
for i in range(len(df)):
    if solution[i] == evidence[i]:
        correct += 1
    total += 1
print(f"The model comes out to be: {correct/total * 100:.2f}% accurate")
