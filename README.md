# CSE 150A---Project
Link to Github: https://github.com/Areco892/CSE-150A---Project.git

Links related to Milestone #2:

Update for Resubmission: (all code aggregated into one single file)

Model #1:               https://github.com/Areco892/CSE-150A---Project/blob/main/Model%201.py

Dataset:                https://github.com/Areco892/CSE-150A---Project/blob/main/Traffic.csv

Credits:                https://github.com/Areco892/CSE-150A---Project/blob/main/CREDITS.txt

--------------------------------------------------------MILESTONE #1---------------------------------------------------------

Abstract

In this project, we will design a traffic control system using a utility-based agent that will help us optimize the flow of traffic. 
The environment would be modeled using an HMM, where previous data will help us predict future traffic movement, allowing us to optimize the flow of traffic. 

Traffic Prediction Datasets:

- https://www.kaggle.com/datasets/hasibullahaman/traffic-prediction-dataset
- https://www.kaggle.com/datasets/fedesoriano/traffic-prediction-dataset

--------------------------------------------------------MILESTONE #2---------------------------------------------------------

Agent Description

The agent is a Utility Based AI Agent, which will try to optimize the flow of traffic.
The agent will have the following parameters (in terms of PEAS):
- Performance measure: minimization of traffic congestion, maximize traffic flow.
- Environment: urban road network.
- Actuators: traffic lights, pedestrian signals.
- Sensors: cameras, public transport data.

For simplicity, there were a lot of assumptions made in the agent's "world". For example, it is assumed the roads are in perfect condition (e.g. no potholes), and also that there are no vehicle accidents. This allows the agent to just focus on how the traffic condition (low traffic, normal traffic, heavy traffic) was based on how many cars were at the intersection/near traffic stops at given time intervals (15 minutes for our data). Note: For convenience, there are only 3 hidden states. The data provided us with 4 traffic situations (low, normal, high, heavy). For this model, we will combine normal and high into just normal. So, the three hidden states for this model are low, normal, heavy.

Agent Setup & Preprocess (Update for Resubmission)

The data provided by the dataset include time, date, day of the week, car count, bike count, bus count, truck count, total, and traffic situation. For our model, we will be interested in two latter columns (total and traffic situation). The traffic situation will serve as our hidden states (low traffic, normal traffic, heavy traffic). We will use the total vehicle counts and traffic situations to train our model. 

In order to get our CPTs, we will like to know what are our transition probabilities and emission probabilities. For our transition probabilities, we are interested in knowing what are the probabilities to transition between low, normal, and heavy traffic. We can do so by taking look at pairs of consecutive data points and then comparing the traffic situation in both cases. For our emission probabilities, we are interested in knowing what are the probabilities of seeing vehicles counts given the current traffic situation. In order to do so, we are going to first set up some ranges (in vehicle counts): (0 - 26) , (26 - 168), (168 - ). These ranges were collected by seeing the range of vehicle counts for each traffic situation and modified accordingly to get a reasonable accuracy. Then for our emission probabilities, we take a look at how many datapoint in each traffic situation fall in those ranges. 

Probabilistic Model

A way we can relate our agent to probabilistic modeling is we are using previous observations (evidence/data collected about how traffic looks like) in order to predict the most probable sequence of traffic conditions (hidden states). So, a probabilistic model that would work great here would be a Hidden Markov Model.

Model, Results, & Conclusion (Update for Resubmission)

In order to train the first model, we first gathered data from the datasets mentioned previously. We then proceed to preprocess and gather all the data pertaining our goal. In this case, we are interested in the total number of vehicles (cars, buses, trucks, bikes) per 15 minutes and the traffic situation at that given time interval. Then, we implemented viterbi's algorithm to find the most probable path given the evidence. Based on different vehicle count ranges, our model would be less or more accurate. So, the ranges for our observed data were modified until the highest possible accuracy was achieved. 

To evaluate our model, we observed how accurate our model predicts a given path. In order to do so, we calculate can Viterbi path accuracy. We get the number of correctly predicted state sequences and divide it by the total number of states. The accuracy for our model was 46.37%. Since different factors usually lead to different traffic conditions, an improvement will be to consider days of the week and the time of the day. For example, people going to work during weekdays greatly affect traffic, as well as, the time of the day (e.g. getting to work or coming from work). Additionally, we can train our model using more diverse dataset / bigger dataset. This will allow to identify potential data skews.

Although far from functional in the real world, the agent gives us an idea of how to optimize the traffic system under idealistic conditions.
