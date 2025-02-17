# CSE 150A---Project
# Link to Github: https://github.com/Areco892/CSE-150A---Project.git

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

Probabilistic Model

A way we can relate our agent to probabilistic modeling is we are using previous observations (evidence/data collected about how traffic looks like) in order to predict the most probable sequence of traffic conditions (hidden states). So, a probabilistic model that would work great here would be a Hidden Markov Model.

Training first model

In order to train the first model, we first gathered data from the datasets mentioned previously. We then proceed to preprocess and gather all the data pertaining our goal. In this case, we are interested in the total number of vehicles (cars, buses, trucks, bikes) per 15 minutes and the traffic situation at that given time interval. Then to train our model, we implement viterbi's algorithm to find the most probable path given the evidence. 

Evaluating first model

To evaluate our model, we observe how accurate our model predicts a given path. In order to do so, we calculate can Viterbi path accuracy. We get the number of correctly predicted state sequences and divide it by the total number of states.

Conclusion

Although far from functional in the real world, the agent gives us an idea of how to optimize the traffic system under idealistic conditions. Since different factors usually lead to different traffic conditions, an improvement will be to consider days of the week and the time of the day. For exapmle, people going to work during weekdays greatly affect traffic, as well as, the time of the day (e.g. getting to work or coming from work).
