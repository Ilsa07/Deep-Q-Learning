import numpy as np
import torch
from collections import deque
import time
import random

class Network(torch.nn.Module):
    """The Network class inherits the torch.nn.Module class, which represents a neural network"""
    def __init__(self, input_dimension, output_dimension):
        """Initialise the network with and input dimension of 2 (x,y coordinate) and output dimension of 3 (move up, down right actions)"""
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers, it has two hidden layers, each with 200 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=200)
        self.layer_2 = torch.nn.Linear(in_features=200, out_features=200)
        self.output_layer = torch.nn.Linear(in_features=200, out_features=output_dimension)

    def forward(self, input):
        """Function which sends some input data through the network and returns the network's output"""
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        return output


class DQN:
    def __init__(self, learning_rate, gamma):
        self.learning_rate = learning_rate
        self.gamma = gamma
        # Create a Q-network and a Target network
        self.q_network = Network(input_dimension=2, output_dimension=3)
        self.target_network = Network(input_dimension=2, output_dimension=3)
        # Define the optimiser which is used when updating the Q-network.
        # The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
    
    def update_target_network(self):
        """Copy the weights of the Q-network to the target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def train_q_network(self, minibatch: list) ->type(float):
        """Function used to train the network, its input is a list containing transactions"""
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss(minibatch)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect 
        # to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        return loss.item()

    def _calculate_loss(self, minibatch: list):
        """Calculate the loss for a list or transitions"""
        # Unpack the elements in the minibatch anc create a list or states, actions,
        # rewards and successor states and convert them to tesnors
        state_array, action_array, reward_array, successor_state_array = zip(*minibatch)
        minibatch_states_tensor = torch.tensor(state_array, dtype = torch.float32)
        minibatch_actions_tensor = torch.tensor(action_array)
        reward_tensor = torch.tensor(reward_array, dtype = torch.float32)
        successor_state_tensor = torch.tensor(successor_state_array, dtype = torch.float32)

        # Get Q(S, a) in a tensor for the number of transitions for the minibatch from the Q-Network
        predicted_q_value_tensor = self.q_network.forward(minibatch_states_tensor).gather(dim=1, index=minibatch_actions_tensor.unsqueeze(-1)).squeeze(-1)
        # Calculate maxaQ(S', a) using the Target Network
        successor_state_q_values = self.target_network.forward(successor_state_tensor)
        successor_state_q_values = torch.amax(successor_state_q_values, 1)
        
        # Calculate the MSE Loss using the Bellman Equation
        gamma_times_successor = torch.mul(successor_state_q_values, self.gamma)
        label = torch.add(reward_tensor, gamma_times_successor)
        loss = torch.nn.MSELoss()(label, predicted_q_value_tensor)
        return loss


class Agent:
    def __init__(self):
        # Initialise the parameters of the agent
        self.gamma = 0.99
        self.initial_epsilon = 0.95
        self.epsilon = self.initial_epsilon
        self.learning_rate = 0.001
        self.minibatch_size = 800
        self.episode_length = 1000
        self.num_steps_taken = 0
        self.episode_count = 0
        self.state = None
        self.action = None
        self.stop_training = False
        self.evaluation_episode = False
        self.start_time = time.time()
        self.buffer = deque(maxlen=20000)
        self.network = Network(input_dimension=2, output_dimension=3)
        self.dqn = DQN(self.learning_rate, self.gamma)

    def has_finished_episode(self):
        """Function to check whether the agent has reached the end of an episode, if it did update the epsisode length based on the amount of time passed"""
        if self.num_steps_taken % self.episode_length == 0:
            # Increase the episode count
            self.episode_count += 1
            # Reset the step count
            self.num_steps_taken = 0
            # Calculate the number of minutes passed of training
            minutes_passed = round((time.time() - self.start_time)/60)
            # Based on how far in the training the agent is, set the episode length.
            # It is decreased every minute, since we do not want to train on transitions that
            # are next to the goal state
            if minutes_passed <10:
                self.episode_length = 1000 - minutes_passed*100
            
            # If the episode is one of the following, set the evaluation flag to true and
            # the episode length to 100 steps
            if self.episode_count in [32, 38, 44, 50, 56, 60, 64, 67, 70, 73]:
                self.evaluation_episode = True
                self.episode_length = 100
            else:
                self.evaluation_episode = False

            return True
        else:
            return False


    # Replay Buffer Functions
    # ***********************
    def get_buffer_current_size(self) -> type(int):
        """Return the number of transitions currently in the buffer"""
        return len(self.buffer)

    def add_transition(self, transition: list) -> type(None):
        """Append a value to the buffer (from the right)"""
        self.buffer.append(transition)

    def sample_minibatch(self, mini_batch_size: int) -> type(list):
        """Sample a random minibatch of transitions from the buffer, always include the last (latest) transition"""
        # Sample mini_batch_size number of elements from the buffer
        return random.choices(self.buffer, k=mini_batch_size)


    # Functions for choosing actions from the network
    # ***********************************************
    def decrease_epsilon(self):
        """Linearly decrease epsilon based on time passed from initial epsilon to 0.02"""
        # Calculate how much time is remaining
        seconds_passed = time.time() - self.start_time

        # Set epsilon based on the amount of time passed
        if self.epsilon > 0.02:
            self.epsilon = self.initial_epsilon - ((self.initial_epsilon - 0.02)/550)*seconds_passed
        #print(self.epsilon)

    def e_greedy_policy(self, state: list) -> type(int):
        """Returns the index of the e-greedy action in a state."""
        # Calculate/decrease epsilon based on the amount of time passed
        self.decrease_epsilon()
        
        # Choose an e-greedy action
        rand_val = np.random.rand(1)
        if self.epsilon < rand_val:
            # If epsilon is smaller than the random variable, return the greedy action index.
            # Epsilon sets the probability of the greedy action, the smaller it is the more
            # likely that the greedy action will be choosen.
            return self.greedy_policy_action(state)
        else:
            # Return a random action
            return random.choice([0,1,2])

    def greedy_policy_action(self, state: list) -> type(int):
        """Return the index of the action with the highest Q value in the input state"""
        # Put the coordinate through the network
        q_values = self.dqn.q_network.forward(torch.tensor(state))
        # Convert the tensor to a list for easier handling
        q_values = q_values.tolist()
        # Get the index of the action with the maximum q value and return it
        return q_values.index(max(q_values))
    
    def _discrete_action_to_continuous(self, discrete_action: int) -> type(list):
        """Convert a discrete action to continuos, each step a movement of 0.02; 0 -> right, 1 -> up, 2 -> down"""
        if discrete_action == 0:
            return np.array([0.02, 0], dtype=np.float32)
        elif discrete_action == 1:
            return np.array([0, 0.02], dtype=np.float32)
        elif discrete_action == 2:
            return np.array([0, -0.02], dtype=np.float32)


    # Functions executed in train_and_test.py
    # ***************************************
    def get_next_action(self, state: list) -> type(list):
        """Function gets called during the training of the agent, returns a continuous e-greedy action in the state or greedy action if the agent is in an evaluation episode"""
        # If it is an evaluation episode, always return the greedy action
        if self.evaluation_episode:
            # Get the e-greedy discrete action
            discrete_action = self.greedy_policy_action(state)
            # Update the number of steps taken, the current state and the current action
            self.num_steps_taken += 1
            self.state = state
            self.action = discrete_action
            return self._discrete_action_to_continuous(discrete_action)
        else:
            # Get the e-greedy discrete action
            discrete_action = self.e_greedy_policy(state)
            # Update the number of steps taken, the current state and the current action
            self.num_steps_taken += 1
            self.state = state
            self.action = discrete_action
            return self._discrete_action_to_continuous(discrete_action)

    def set_next_state_and_distance(self, next_state: list, distance_to_goal: float) -> type(None):
        """Function gets called after every step to give information to the agent on the transition"""
        # Convert the distance of the goal to a reward, discount it if it bumped into a wall (state=next state)
        if self.state[0] == next_state[0] and self.state[1] == next_state[1]:
            reward = ((1 - distance_to_goal)**2)*0.6
        else:
            reward = ((1 - distance_to_goal)**2)
        
        # Create a transition and add it to the buffer
        transition = (self.state, self.action, reward, next_state)
        self.add_transition(transition)

        # If it is an evaluation episode and you reached the goal stop training
        if self.evaluation_episode and distance_to_goal < 0.03:
            self.stop_training = True

        # If it is an evaluation episode, dont save the transitions or train the network
        if self.evaluation_episode:
            return

        # If the stop training flag is true, dont train the agent anymore
        if self.stop_training:
            return

        # If there are enough elements in the buffer to get a minibatch
        if self.get_buffer_current_size() > self.minibatch_size:
            # Sample the buffer and train the Network
            mini_batch = self.sample_minibatch(self.minibatch_size)
            self.dqn.train_q_network(mini_batch)
       
        # Update the Target Network every 50 steps
        if (self.num_steps_taken % 50 ==0):
            self.dqn.update_target_network()

    def get_greedy_action(self, state: list) -> type(list):
        """Function that gets called during the evaluation of the agent, always return a continuous greedy action."""
        # Get the index of the greedy action in the input state
        action = self.greedy_policy_action(state)
        # Convert it to a continuous action and return int
        return self._discrete_action_to_continuous(action)
