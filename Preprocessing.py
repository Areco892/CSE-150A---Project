import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
Data Exploration and Preprocessing
"""
def preprocess(df):
    # Number of observations
    def number_of_observations(df):
        print(f"- There are {len(df)} observations")

    # Data Distribution and Outlier Analysis (Numerical Columns)
    """
    Since we will only be interested on the total vehicle count per 15 minutes, 
    we can just observed how is the data from this column distrubuted.
    """
    def data_distribution(df):
        print(f'\n- Distribution of Total Vehicle per 15 minutes: ')
        plt.figure(figsize=(10, 4))
        sns.histplot(df["Total"], kde=True, bins=30)
        plt.title(f'Distribution of Total Vehicle per 15 minutes')
        plt.show()

    # Scales of Features: Check for skewness and normalization requirements
    def feature_scales(df):
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        skewness = df[numeric_columns].skew()
        print("\n- Skewness of Numeric Columns:")
        print(skewness)
        
        # Identifying if scaling might be necessary
        print("Columns with skewness > 1 (might require normalization):")
        print(skewness[skewness > 1])

    # Missing Data Analysis
    def missing_data(df):
        missing = df.isnull().sum()
        missing_percentage = (missing / len(df)) * 100
        missing_data = pd.DataFrame({'Missing Values': missing, 'Percentage': missing_percentage})
        print("\n- Missing Data Info:")
        print(missing_data[missing_data['Missing Values'] > 0].sort_values(by='Percentage', ascending=False))

    # Column descriptions
    def data_overview(df):
        print("\n- Data Shape (rows x columns):", df.shape)  # number of rows and columns
        print("\n- Data Info and Column Desciptions:")
        df.info()  # General info about data types and memory usage

    # Running the data exploration functions
    print("Data Exploration and Preprocessing\n")
    number_of_observations(df)
    data_distribution(df)
    feature_scales(df)
    missing_data(df)
    data_overview(df)


"""
Convert Observations into Probabilities and Generate CPTs
"""
def getTransitionMatrix(df):
    """
    Transition matrix: from current state -> next state
    Hidden States = [low, normal, heavy]

        [P(Low|Low)      P(Normal|Low)      P(High|Low)
         P(Low|Normal)   P(Normal|Normal)   P(High|Normal)
         P(Low|Heavy)    P(Normal|Heavy)    P(High|Heavy)]

        For our training, our transition matrix is:
                    Low        Normal      Heavy
        Low     [[0.3125     0.65131579 0.03618421]
        Normal   [0.10055304 0.82151835 0.07792861]
        Heavy    [0.01173021 0.23167155 0.75659824]]
    """
    # Create a 3x3 array to represent the transition matrix
    total_entries = len(df)
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
    return trans_matrix

"""
To get the range for our observations, we can take a look at the 
minimum and maximum vehicle count on every traffic situation

min = 999
max = 0
for i in range(total_entries):
    if df.iloc[i]["Traffic Situation"] == "___": # "___" <- [low / normal or high / heavy]
        curr_observation = df.iloc[i]["Total"]
        if curr_observation > max:
            max = curr_observation
        if curr_observation < min:
            min = curr_observation
#print(min)
#print(max)

After running the above, we get our range for Observations:
small count: 21 - 111
moderate count: 26 - 223
high count: 168 - 279
"""

def getEmissionMatrix(df):
    """
    Emission matrix: from current state -> current observation
    
    Based on the range of our observations, for the emission matrix, 
    we will define the following observations Vehicle count per unit time. 
    So, these will be our observations.
    count_small:    0 - 26
    count_medium:  26 - 168
    count_high:   168 - above

    Observations: [A (small vehicle count)    (< 26), 
                   B (moderate vehicle count) (< 168), 
                   C (high vehicle count)     (> 168)]

              small       moderate       high
    low     [P(A|Low)     P(B|Low)     P(C|Low)   
    normal   P(A|Normal)  P(B|Normal)  P(C|Normal)    
    heavy    P(A|Heavy)   P(B|Heavy)   P(C|Heavy) ]

    For our traning, our emission matrix is
                   small      moderate    high
    low        [[1.         0.         0.        ]
    normal      [0.12604672 0.87395328 0.        ]
    heavy       [0.         0.01015965 0.98984035]]
    """
    # Create a 3x3 array to represent the emission matrix
    total_entries = len(df)
    emission_matrix = np.zeros((3,3))

    # In order to get the CPTs, we will take a look at the amount of
    # vehicles on every traffic situation (low, normal, heavy).
    for i in range(total_entries):
        curr_state = df.iloc[i]["Traffic Situation"]
        curr_count = df.iloc[i]["Total"]
        if curr_count < 26:
            if curr_state == "low":
                emission_matrix[0][0] += 1
            elif curr_state == "normal" or curr_state == "high":
                emission_matrix[0][1] += 1
            elif curr_state == "heavy":
                emission_matrix[0][2] += 1
        elif curr_count < 168:
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

    # We can get our CPT's by dividing by the total amount of vehicles on 
    # every traffic situation.
    count = 0
    for i in range(3):
        count = emission_matrix[i][0] + emission_matrix[i][1] + emission_matrix[i][2]
        emission_matrix[i][0] = emission_matrix[i][0] / count
        emission_matrix[i][1] = emission_matrix[i][1] / count
        emission_matrix[i][2] = emission_matrix[i][2] / count
        count = 0
    
    return emission_matrix
