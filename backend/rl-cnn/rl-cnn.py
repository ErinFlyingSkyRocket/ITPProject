import tensorflow as tf
tf.config.run_functions_eagerly(True)  # Version Compatibility

import numpy as np
from cnn_architecture_design import conv_layers, pool_layers, dense_layers, optimizers, generate_cnn_model, generate_selective_actions, dropout_layers
import os, sys, json

from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.layers import Dense, LSTM
import random
import pickle
import optuna

def optuna_objective(trial):
    # Sample architecture components using trial
    state = (
        trial.suggest_int("conv1", 0, len(conv_layers)-1),
        trial.suggest_int("pool1", 0, len(pool_layers)-1),
        trial.suggest_int("dropout1", 0, len(dropout_layers)-1),
        trial.suggest_int("conv2", 0, len(conv_layers)-1),
        trial.suggest_int("pool2", 0, len(pool_layers)-1),
        trial.suggest_int("dropout2", 0, len(dropout_layers)-1),
        trial.suggest_int("dense", 0, len(dense_layers)-1),
        trial.suggest_int("dropout3", 0, len(dropout_layers)-1),
        trial.suggest_int("optimizer", 0, len(optimizers)-1),
    )
    model = generate_cnn_model(state)
    model.fit(train_images, train_labels, batch_size=128, epochs=5, verbose=0)
    _, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    return test_acc  # Maximize accuracy


class DQN():
    def __init__(self, state_size, action_size, filename=None):
        self.batch_size = 32
        self.state_size = state_size
        self.action_size = action_size
        self.discount_factor = 0.95

        self.model = self.create_model()
        self.memory = deque(maxlen=50000)

    def create_model(self):
        """
        neural network model
        :return:
        """
        model = Sequential()
        model.add(Dense(20, input_shape=(self.state_size,), activation="relu"))
        model.add(Dense(15, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(optimizer=Adam(), loss="mse")

        return model

    def replay(self):
        """
        experience replay. find the q-value and train the neural network model with state as input and q-values as targets
        :return:
        """
        batch = random.choices(self.memory,k=self.batch_size)
        print("Fit the model individually using each of the 32 samples")

        for state, next_state, index, reward, done in batch:
            max_future_q = np.max(self.model.predict(np.array([next_state]))[0])
            new_q = reward + self.discount_factor * max_future_q
            current_q = self.model.predict(np.array([state]))
            current_q[0][index] = new_q
            self.model.fit(np.array([state]), current_q)

def padding_reshape(data):
    num_input = 62
    cnn_input_shape = 81
    num_padding = 81 - 62
    left_num_padding = num_padding // 2
    right_num_padding = num_padding - left_num_padding
    padded_data = np.pad(data, ((0, 0), (left_num_padding, right_num_padding)), mode='constant', constant_values=0)
    padded_data = padded_data.reshape((-1, 9, 9))
    padded_data = np.expand_dims(padded_data, axis=-1)
    return padded_data

def save_to_txt(content):
    with open('output.txt', 'a') as f:
        f.write(content)
        f.write('\n')

data_train = np.load("data_train.npy")
data_test = np.load("data_test.npy")
train_labels = np.load("int_label_train.npy")
test_labels = np.load("int_label_test.npy")

train_images = padding_reshape(data_train[:,:62])
test_images = padding_reshape(data_test[:,:62])

study = optuna.create_study(direction="maximize")
study.optimize(optuna_objective, n_trials=100)  # Tune number of trials if needed

optuna_best_state = (
    study.best_params["conv1"],
    study.best_params["pool1"],
    study.best_params["dropout1"],
    study.best_params["conv2"],
    study.best_params["pool2"],
    study.best_params["dropout2"],
    study.best_params["dense"],
    study.best_params["dropout3"],
    study.best_params["optimizer"],
)
print("🎯 Best Optuna state:", optuna_best_state)


episodes = 50
num_steps = 50
reward_func = 3


# Exploration settings
epsilon = 1  # not a constant, qoing to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = episodes//2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

epsilon_values = [epsilon]
for episode in range(episodes):
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value
    epsilon_values.append(epsilon)


# Goal for the CNN model to reach
target_test_acc = 0.99 # 1.0
target_test_acc_filename = "target_test_acc.npy"

# Collect rewards across the episode
ep_rewards = []

# Structure of CNN architecture is 
# 1. conv_layer1
# 2. pool_layer1
# 3. dropout_layer1
# 4. conv_layer2
# 5. pool_layer2
# 6. dropout_layer2
# 7. flatten 
# 8. dense_layer
# 9. dropout_layer3
# 10. output layer

# Number of convolutional, pool, dropout layers in CNN architecture
num_conv_layer = 2

actions = generate_selective_actions(num_conv_layer)

state_sizes = [len(conv_layers), len(pool_layers), len(dropout_layers), len(conv_layers), len(pool_layers), len(dropout_layers), len(dense_layers), len(dropout_layers), len(optimizers)]


# Create a new model
agent = DQN(len(state_sizes), len(actions), None)

for episode in range(episodes):
    epsilon = epsilon_values[episode]

    # When RL agent advance from exploration phrase to exploitation phrase
    if epsilon <= 0:
        if os.path.isfile(target_test_acc_filename):
            target_test_acc = np.load(target_test_acc_filename)

    print("episode", episode, "epsilon", epsilon, "target_test_acc", target_test_acc)

    # Flag to prevent from trying endlessly to generate a new state or an action 
    flag = True
    flag_count = 0
    # Max number of times generating a new default state or an action to create a new state during iteration before stopping the program
    max_tries = 10

    episode_reward = 0

    # Collect all states throughout an iteration loop for an episode
    iteration_states = []
    # Collect accuracy for all states throughout an iteration loop for an episode
    iteration_test_acc = []

    # Initial state
    #state = tuple([1, 0, 6, 3, 0, 1, 3, 4, 7])
    state = optuna_best_state

    # Previous test_acc, not previous best test_acc
    prev_test_acc = 0


    step = 0
    done = False

    # Entering the iteration Loop
    while not done and step < num_steps and flag:
        model = generate_cnn_model(state)

        str_state = ",".join([str(value) for value in state])

        # Train the CNN model
        model.fit(train_images, train_labels, batch_size=128, epochs=5)

        # Get the test accuracy
        _, test_acc = model.evaluate(test_images,  test_labels, verbose=0)

        if reward_func == 3:
            # Reward function 3 to get the reward
            if (test_acc - prev_test_acc) >= 0.05:
                # current accuracy is more than previous accuracy by 5% or more
                reward = test_acc + 5*test_acc
            elif test_acc > prev_test_acc:
                # current accuracy is more than previous accuracy but the difference is less than 5%
                reward = test_acc + 2*test_acc
            elif test_acc < prev_test_acc:
                reward = -(pow((1+(prev_test_acc/test_acc)), 2))

            if test_acc > target_test_acc:
                done = True

        prev_test_acc = test_acc

        episode_reward += reward

        print(state, 'Test accuracy:', test_acc, reward)

        if step == 0:
            first_state = state
            first_test_acc = test_acc

        iteration_states.append(state)
        iteration_test_acc.append(test_acc)

        new_state_value = False
        flag_count = 0
        print("Going into loop to generating an action for a new state")
        while not new_state_value and flag and not done:
            if np.random.random() > epsilon:
                # Exploit
                # # Get action from DQN
                index = np.argmax(agent.model.predict(np.array([state])))
            else:
                # Explore
                # Get random action
                index = np.random.randint(0, len(actions))

            # Get the action from the list of actions
            action = actions[index]

            # Check that the current value in that state is not the same value as the value used to replace it
            if action[1] != state[action[0]]:
                new_state = list(state)
                new_state[action[0]] = action[1]
                new_state = tuple(new_state)
                #  Check that this new_state is also not already in iteration_states
                if not new_state in iteration_states:
                    new_state_value = True
                else:
                    pass
            flag_count += 1
            if flag_count > max_tries:
                print("episode", episode, "step", step, "flag_count",flag_count, "Exit out of program if unable to generate an action that can provide a new state that has not been generated before")
                flag = False
                break

        # Checked Target accuracy has been reached
        if done:
            if np.random.random() > epsilon:
                # Exploit
                # Get action from DQN
                index = np.argmax(agent.model.predict(np.array([state])))
            else:
                # Explore
                # Get random action
                index = np.random.randint(0, len(actions))
                print("Explore")

            # Get the action from the list of actions
            action = actions[index]
            new_state = list(state)
            new_state[action[0]] = action[1]
            new_state = tuple(new_state)

        if flag or done:
            agent.memory.append((state, new_state, index, reward, done))


        if flag and not done:
            state = new_state

        step += 1


    # Experience replay at the end of each episode to update the model using the states and its q-values
    agent.replay()

    best_state = iteration_states[iteration_test_acc.index(max(iteration_test_acc))]
    best_state_str = ",".join([str(value) for value in best_state])

    # # episode
    # # number of iterations completed
    # # highest accuracy achieved during this episode
    # # state for the highest accuracy achieved during this episode
    # # starting accuracy achieved during this episode
    # # state for the starting accuracy achieved during this episode
    # # last accuracy achieved during this episode
    # # state for the last accuracy achieved during this episode
    # # sum of the rewards for this episode
    # # Has the target accuracy been reached during this episode
    # # target accuracy during this episode
    print_statement = str(episode) + "|" + str(len(iteration_states)) + "|" + str(max(iteration_test_acc)) + "|" + best_state_str + "|"  + str(first_test_acc) + "|" + ",".join([str(value) for value in first_state]) + "|"  + str(test_acc) + "|" + ",".join([str(value) for value in state]) + "|" + str(episode_reward) + "|" + str(target_test_acc)

    print(print_statement)
    save_to_txt(print_statement)



    episode_max_test_acc = max(iteration_test_acc)
    if os.path.isfile(target_test_acc_filename):
        stored_target_test_acc = np.load(target_test_acc_filename)
        # Check if accuracy obtain during this episode is higher than the target test accuracy for exploitation phrase
        if episode_max_test_acc > stored_target_test_acc:
            np.save(target_test_acc_filename, episode_max_test_acc)
    else:
        np.save(target_test_acc_filename, episode_max_test_acc)


def extract_best_state_from_output(file_path):
    """
    Reads the output.txt log and extracts the state with the highest test accuracy.
    Returns the best state as a tuple of integers.
    """
    best_acc = -1
    best_state = None

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) < 4:
                continue
            try:
                acc = float(parts[2])
                state_str = parts[3]
                state = tuple(map(int, state_str.split(',')))
                if acc > best_acc:
                    best_acc = acc
                    best_state = state
            except:
                continue

    return best_state


# === Save final best model after training ===
final_best_state = extract_best_state_from_output('output.txt')  # define this function below
print("🎯 Final best state:", final_best_state)

final_model = generate_cnn_model(final_best_state)
final_model.fit(train_images, train_labels, batch_size=128, epochs=5)
final_model.save("final_model.h5")
print("✅ Final model saved as final_model.h5")


