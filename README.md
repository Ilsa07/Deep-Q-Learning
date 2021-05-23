# Deep-Q-Learning
The following project is an implementation of a **Deep Q Learning** algorithm, which was developped as an assignment at Imperial College London. The environment in the random_environment.py file was provided to us by the college. The following algorithm solves a random generated maze by training for 10 minutes and then doing a test run.


## Detailed Description of the Algorithm
I have implemented a Deep Q-learning algorithm, which relies on the use of choosing epsilon-greedy actions, an experience replay buffer for training on random mini-batches of transitions, and on the use of a target network similar to the one in Part 1 of the coursework but with dynamically set parameters during the testing phase.

A Network class is implemented (7-23), which initialises a neural network with 2 hidden layers, each with 200 units. A DQN (Deep Q Network) class (26-75) initialises the Target and Q-network (31, 32) and interacts with them. A function updates the Target network with the weights of the Q-network (37-39) and is called after every 50 steps (242, 243). A function (55-75) calculates and returns the loss for a mini batch of transitions using the Bellman equation.

The agent has 3 actions, move right, up, or down. The Q and Target networks’ output dimensions were set to 3 (line 31, 32, 95, 168). The magnitudes and indexes of the actions are defined in a function (79-86), each action has a magnitude of 0.02. The reward function (214-217) is (1-x)2 where x is the distance to the goal. Line 214 determines if the agent hit a wall. If it has, the reward is decreased by 40%.

The Agent class is initialised (79-97) and the learning rate was set to α=0.001. A discount factor of γ=0.99 was chosen, since the algorithm has to prioritise the long term reward on the more difficult maps. The exploration parameter is initialised to E = 0.95, but it is decreased linearly to 0.02 based on the amount of time passed (144- 152) and is recalculated before every E-greedy step (157). This was implemented so the agent explores its environment at the beginning of training, but chooses and trains on the greedy actions by the end. The episode length is initially 1000 steps (line 86), but is linearly decreased to 100 based on the number of minutes passed (106-111). Moving around the goal state due to an excess of steps fills the replay buffer with transitions next to the goal state, which the agent should not be trained on to avoid biases. Since both E and the episode length decrease linearly, on shorter episodes more greedy actions will be performed which provide good training data. The Agent class contains a replay buffer with a maximum of 20000 elements (94). A method returns the number of elements currently in the buffer (128-130), and is called (236) to check if the buffer has enough elements to be sampled. A method adds transitions to the buffer (32-34) every time the agent performs a step (221). A method samples the replay buffer randomly and returns a list containing transitions on which the agent will be trained. The mini batch size was set to 800 (85).

Evaluation episodes test if the agent can reach the goal with the current greedy policy in 100 steps and to stop training if it can. During this the  agent  is  not trained, but transitions are added to the buffer. The Boolean flags, ”stop training” and ”evaluation episode”, are initially False (91, 92). At the end of each episode, the algorithm checks if the next episode is an evaluation episode, if yes, it sets the evaluation episode flag to True and the episode length to 100 (115-119). Evaluation episodes are specified by episode number (115) and test the performance of the agent during the last half of the training. After each step the algorithm checks if the distance to the goal is less than 0.03 and if it is the ”stop training” flag is set to True (224, 225), and the agent is no longer trained (232, 233).



## Getting started
1. Clone the project and create a virtual environment
2. Install the required packages in the virtual environment
   ```
   pip3 install -r requirements.txt
   ```
3. Run the algorithm with the train_and_test.py function
