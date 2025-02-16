import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import HMM as HMM

df = pd.read_csv("Traffic.csv")

# Number of entries
total_entries = len(df)

"""
# Get File Information
df.info()
"""

"""
Transition matrix: from current state -> next state
Hidden States = [low, normal, heavy]

[P(Low|Low)      P(Normal|Low)      P(High|Low)
 P(Low|Normal)   P(Normal|Normal)   P(High|Normal)
 P(Low|Heavy)    P(Normal|Heavy)    P(High|Heavy)]

           Low        Normal     Heavy
Low     [[0.3125    0.59210526 0.09539474]
Normal  [0.10851319 0.74340528 0.14808153]
Heavy   [0.02691924 0.24825523 0.72482552]]
"""

# Create a 3x3 array to represent the transition matrix
trans_matrix = np.zeros((3,3))

# Get the count for each of the array entries
for i in range(total_entries - 1):
    curr_state = df.iloc[i]["Traffic Situation"]
    next_state = df.iloc[i+1]["Traffic Situation"]
    
    if curr_state == "low":
        if next_state == "low":
            trans_matrix[0][0] += 1
        elif next_state == "normal" or next_state == "high":
            trans_matrix[0][1] += 1
        else:
            trans_matrix[0][2] += 1
    elif curr_state == "normal" or curr_state == "high":
        if next_state == "low":
            trans_matrix[1][0] += 1
        elif next_state == "normal" or next_state == "high":
            trans_matrix[1][1] += 1
        else:
            trans_matrix[1][2] += 1
    else:
        if next_state == "low":
            trans_matrix[2][0] += 1
        elif next_state == "normal" or next_state == "high":
            trans_matrix[2][1] += 1
        else:
            trans_matrix[2][2] += 1

# Get the probabilities by dividing over the corresponding counts
count = 0
for i in range(3):
    count = trans_matrix[i][0] + trans_matrix[i][1] + trans_matrix[i][2]
    trans_matrix[i][0] = trans_matrix[i][0] / count
    trans_matrix[i][1] = trans_matrix[i][1] / count
    trans_matrix[i][2] = trans_matrix[i][2] / count
    count = 0

"""
min = 999
max = 0
for i in range(total_entries):
    if df.iloc[i]["Traffic Situation"] == "___":
        curr_observation = df.iloc[i]["Total"]
        if curr_observation > max:
            max = curr_observation
        if curr_observation < min:
            min = curr_observation
#print(min)
#print(max)

Range for Observations:
small count: 21 - 111
moderate count: 26 - 223
high count: 168 - 279

Vehicle count per unit time: This will be our observations.
count_small:    0 - 100
count_medium: 100 - 200
count_high:   200 - 
"""

"""
Emission matrix: from current state -> current observation
Observations: [A (small vehicle count  / hour), 
               B (moderate vehicle count / hour), 
               C (high vehicle count   / hour)]

[P(A|Low)    P(B|Low)    P(C|Low)   
 P(A|Normal) P(B|Normal) P(C|Normal)    
 P(A|High)   P(B|High)   P(C|High)  

             small      moderate    high
low      [[0.19471698 0.80528302 0.        ]
normal   [0.03425168  0.6820551  0.28369322]
heavy    [0.          0.02272727 0.97727273]]
"""
emission_matrix = np.zeros((3,3))

for i in range(total_entries):
    curr_state = df.iloc[i]["Traffic Situation"]
    curr_count = df.iloc[i]["Total"]
    if curr_count < 100:
        if curr_state == "low":
            emission_matrix[0][0] += 1
        elif curr_state == "normal" or curr_state == "high":
            emission_matrix[0][1] += 1
        elif curr_state == "heavy":
            emission_matrix[0][2] += 1
    elif curr_count < 200:
        if curr_state == "low":
            emission_matrix[1][0] += 1
        elif curr_state == "normal" or curr_state == "high":
            emission_matrix[1][1] += 1
        elif curr_state == "heavy":
            emission_matrix[1][2] += 1
    else:
        if curr_state == "low":
            emission_matrix[2][0] += 1
        elif curr_state == "normal" or curr_state == "high":
            emission_matrix[2][1] += 1
        elif curr_state == "heavy":
            emission_matrix[2][2] += 1

count = 0
for i in range(3):
    count = emission_matrix[i][0] + emission_matrix[i][1] + emission_matrix[i][2]
    emission_matrix[i][0] = emission_matrix[i][0] / count
    emission_matrix[i][1] = emission_matrix[i][1] / count
    emission_matrix[i][2] = emission_matrix[i][2] / count
    count = 0