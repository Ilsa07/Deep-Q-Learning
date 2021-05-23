# Import some modules from other libraries
import numpy as np
import torch
import time
import random
from collections import deque
from matplotlib import pyplot as plt
import statistics


# Import the environment module
from environment import Environment
from q_value_visualiser import QValueVisualiser


# The Agent class allows the agent to interact with the environment.
class Agent:

    # The class initialisation function.
    def __init__(self, environment):
        # Set the agent's environment.
        self.environment = environment
        # Create the agent's current state
        self.state = None
        # Create the agent's total reward for the current episode.
        self.total_reward = None
        # Reset the agent.
        self.reset()

    # Function to reset the environment, and set the agent to its initial state. This should be done at the start of every episode.
    def reset(self):
        # Reset the environment for the start of the new episode, and set the agent's state to the initial state as defined by the environment.
        self.state = self.environment.reset()
        # Set the agent's total reward for this episode to zero.
        self.total_reward = 0.0

    # Function to make the agent take one step in the environment.
    def step(self, discrete_action):
        # Convert the discrete action into a continuous action.
        continuous_action = self._discrete_action_to_continuous(discrete_action)
        # Take one step in the environment, using this continuous action, based on the agent's current
        # state. This returns the next state, and the new distance to the goal from this new state. 
        # It also draws the environment, if display=True was set when creating the environment object.
        next_state, distance_to_goal = self.environment.step(self.state, continuous_action)
        # Compute the reward for this paction.
        reward = self._compute_reward(distance_to_goal)
        # Create a transition tuple for this step.
        transition = (self.state, discrete_action, reward, next_state)
        # Set the agent's state for the next step, as the next state from this step
        self.state = next_state
        # Update the agent's reward for this episode
        self.total_reward += reward
        # Return the transition
        return transition


    # Function to convert discrete action (as used by a DQN) to a continuous action (as used by the environment).
    def _discrete_action_to_continuous(self, discrete_action):
        if discrete_action == 0:
            # Right
            continuous_action = np.array([0.1, 0], dtype=np.float32)
        elif discrete_action == 1:
            # Up
            continuous_action = np.array([0, 0.1], dtype=np.float32)
        elif discrete_action == 2:
            # LEFT
            continuous_action = np.array([-0.1, 0], dtype=np.float32)
        elif discrete_action == 3:
            # DOWN
            continuous_action = np.array([0, -0.1], dtype=np.float32)

        return continuous_action

    # Function for the agent to compute its reward. In this example, the reward is based on the agent's
    # distance to the goal after the agent takes an action.
    def _compute_reward(self, distance_to_goal):
        reward = float(0.1*(1 - distance_to_goal))
        return reward


# The Network class inherits the torch.nn.Module class, which represents a neural network.
class Network(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 100 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=100)
        self.layer_2 = torch.nn.Linear(in_features=100, out_features=100)
        self.output_layer = torch.nn.Linear(in_features=100, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        return output


# The DQN class determines how to train the above neural network.
class DQN:

    # The class initialisation function.
    def __init__(self, learning_rate, gamma, epsilon):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=4)
        self.target_network = Network(input_dimension=2, output_dimension=4)
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        #self.update_target_network()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def return_q_values_for_plotting(self):
        # Create a list of states for plotting
        q_values = np.zeros((10, 10, 4))

        for col in range(10):
            for row in range(10):    
                x = (col / 10.0) + 0.05
                y = (row / 10.0) + 0.05
                q_value = self.q_network.forward(torch.tensor([x, y]))
                #q_column_values.append(q_value.tolist())
                q_values[col, row] = q_value.tolist()

        return q_values


    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_network(self, minibatch):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss(minibatch)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        return loss.item()


    # Function to calculate the loss for a particular transition.
    def _calculate_loss(self, minibatch):
        state_array, action_array, reward_array, successor_state_array = zip(*minibatch)
        
        minibatch_states_tensor = torch.tensor(state_array, dtype = torch.float32)
        minibatch_actions_tensor = torch.tensor(action_array)
        reward_tensor = torch.tensor(reward_array, dtype = torch.float32)
        successor_state_tensor = torch.tensor(successor_state_array, dtype = torch.float32)

        # This is Q(S, a) in a tensor for the number of transitions for the minibatch
        predicted_q_value_tensor = self.q_network.forward(minibatch_states_tensor).gather(dim=1, index=minibatch_actions_tensor.unsqueeze(-1)).squeeze(-1)
        # This is maxaQ(s', a)
        successor_state_q_values = self.target_network.forward(successor_state_tensor)
        successor_state_q_values = torch.amax(successor_state_q_values, 1)
        
        gamma_times_successor = torch.mul(successor_state_q_values, self.gamma)
        label = torch.add(reward_tensor, gamma_times_successor)
        loss = torch.nn.MSELoss()(label, predicted_q_value_tensor)
        return loss


    def greedy_policy_action(self, state):
        """
        state should be an array in the form of [x_coordinate, y_coordinate]
        """
        # Put the coordinate through the network
        q_values = self.q_network.forward(torch.tensor(state))
        # Get the index of the action with the maximum q value
        q_values = q_values.tolist()

        greedy_action_index = q_values.index(max(q_values))

        return greedy_action_index
    

    # You want to return an index in this case
    def e_greedy_policy(self, state):
        # Decrease the value of epsolon at every step
        if self.epsilon > 0.02:
            self.epsilon = self.epsilon*0.9995
        else:
            pass
        
        # Choose an e-greedy action
        rand_val = np.random.rand(1)
        if self.epsilon < rand_val:
            return self.greedy_policy_action(state)
        else:
            return np.random.choice(4)
        

class ReplayBuffer:
    def __init__(self):
        self.buffer = deque(maxlen=5000)
    
    def get_buffer_current_size(self):
        return len(self.buffer)

    def add_transition(self, transition):
        """Append a value to the right of the buffer"""
        self.buffer.append(transition)

    def sample_minibatch(self, mini_batch_size):
        """Sample a random minibatch of transitions from the replay buffer"""
        # return a random mini patch
        return random.choices(self.buffer, k=mini_batch_size)


if __name__ == "__main__":
    environment = Environment(display=True, magnification=500)
    agent = Agent(environment)
    visualiser = QValueVisualiser(environment=environment, magnification=500)
    dqn = DQN(learning_rate=0.001, gamma=0.9, epsilon=0.9)
    buffer = ReplayBuffer()


    fig, ax = plt.subplots()
    ax.set(xlabel='Episodes', ylabel='Loss', title='Loss Curve for Target Network')
    losses = []
    iterations = []


    # Loop over episodes
    for episode_number in range(200):
        # Reset the environment for the start of the episode.
        agent.reset()
        step_losses = []

        # Loop over steps within this episode. The episode length here is 20.
        for step_num in range(50):
            discrete_action = dqn.e_greedy_policy(agent.state)
            transition = agent.step(discrete_action)
            buffer.add_transition(transition)

            if buffer.get_buffer_current_size() < 100:
                pass
            else:
                mini_batch = buffer.sample_minibatch(100)
                # Train on this mini batch and get loss
                loss = dqn.train_q_network(mini_batch)
                # Append Loss for plotting
                step_losses.append(loss)
        
        # Update the agent every 10 episodes
        if (episode_number % 5 ==0):
            dqn.update_target_network()

        # Record the average loss over the trace
        if len(step_losses) != 0:
            losses.append(sum(step_losses)/len(step_losses))
            iterations.append(episode_number)
        
        # Visualise Q values
        q_values = dqn.return_q_values_for_plotting()
        visualiser.draw_q_values(q_values)
    

    trained_transitions = []
    # Run once with the trained Q network and visualise shit
    agent.reset()
    for step_num in range(20):
        # Get the action corresponding to the greedy policy
        discrete_action = dqn.e_greedy_policy(agent.state)
        # [state, action, reward, next state]
        transition = agent.step(discrete_action)
        trained_transitions.append(transition)
        #loss = dqn.train_q_network(transition)
    
    states = []
    for step in trained_transitions:
        states.append(step[0])

    environment.draw_path(states)
        
    print(f'The variance of the loss dataset is: {statistics.variance(losses)}')
    # Plot and save the loss vs iterations graph
    ax.plot(iterations, losses, color='blue')
    plt.yscale('log')
    #plt.show()
    fig.savefig("loss_vs_iterations_in_minibatch.png")
