# -*- coding: utf-8 -*-
"""
@author: Predictive Analysts
"""

import gym
import numpy as np
import time
import random
from sklearn.neural_network import MLPClassifier
import copy

import matplotlib.pyplot as plt


# new class
class MLPClassifierOverride(MLPClassifier):
    # Overriding _init_coef method
    def _init_coef(self, fan_in, fan_out):
        if self.activation == 'logistic':
            init_bound = np.sqrt(2. / (fan_in + fan_out))
        elif self.activation in ('identity', 'tanh', 'relu'):
            init_bound = np.sqrt(6. / (fan_in + fan_out))
        else:
            raise ValueError("Unknown activation function %s" %
                             self.activation)
        coef_init = [[0.54804146, 0.54804146, 0.54804146],
                     [0.54804146, -0.54804146, 0.54804146],
                     [0.54804146, 0.54804146, 0.54804146],
                     [0.54804146, 0.54804146, 0.54804146]]
        ### place your initial values for coef_init here

        intercept_init = [[-0.54804146, 0.70052231 , -0.87538128]]### place your initial values for intercept_init here
        return coef_init, intercept_init



def initial_population(population_size, hlayer_size):
    """
    Create an initial population of ANN classifiers
    """
    ann_population = []  # List for ANN population
    for ann in range(population_size):
        mlp = MLPClassifierOverride(batch_size=1, max_iter=1, solver='sgd',
                            activation='relu', learning_rate='invscaling',
                            hidden_layer_sizes=hlayer_size, random_state=1)

        mlp.partial_fit(np.array([env.observation_space.sample()], dtype='int64'),
                        np.array([env.action_space.sample()], dtype='int64'),
                        classes=np.arange(env.action_space.n))
        # mlp.coefs_[0]
        print('=========================================')
        print(mlp.intercepts_[0])
        print('=========================================')

        ann_population.append(mlp)
    return ann_population


def population_score(ann_population, iterations, time_between, delay):
    """
    Calculate the score of each ANN in the population (generation)
    """
    i = 0
    ann_rewards = np.zeros(len(ann_population))
    for mlp in ann_population:
        iteration = 0
        total_reward = 0
        env.reset()
        action = 1
        # action = random.sample(set(np.arange(env.action_space.n)), 1)[0]
        # same_action = [action, 1] # Sanity check
        action_sequence = np.zeros(1000)
        observation, reward, done, info = env.step(action)  # Initial random action

        while not done and iteration < iterations:
            total_reward += reward
            np.put(action_sequence, [iteration], [action])
            action = mlp.predict(observation.reshape(1, -1))[0]  # Reshaped data due
            # action_sequence[iteration] = action
            iteration += 1
            #
            if all(i == action for i in action_sequence[iteration - 4:iteration]) and iteration >= 4:
                # print('=========================================')
                action_flip = 1 if action == 0 else 0
                action = action_flip
                # print(action_sequence)
                # print('Same')
                # print('=========================================')

            # to 2aD array expected
            # same_action[1] += (action == same_action[0])
            # same_action[0] = action
            # if same_action[1] == 5: # Five consecutive same actions
            #     action = int(not bool(action)) # Possible actions are only 0 an 1.
            #     # Take the opposite action
            #     same_action = [action, 1]
            observation, reward, done, info = env.step(action)
            # env.render()
            # time.sleep(delay)
            mlp.partial_fit(observation.reshape(1, -1), np.array([action]))
        ann_rewards[i] = total_reward
        i += 1
    return ann_rewards


def find_parents(ann_rewards):
    """
    Calculate the probabilities of ANNs and find parents ordering.
    Find best and average score.
    """
    best_score = max(ann_rewards)
    average_score = ann_rewards.mean()
    ann_prob = ann_rewards / sum(ann_rewards)
    parents_idx = np.argsort(ann_prob)
    parent1 = ann_population[parents_idx[0]]
    return parents_idx, parent1, best_score, average_score


def breed_swap_switch(parent1, parent2):
    """
    Crossover: swap and switch applied on:
    Input level weights
    Hidden level weights
    Input level biases
    Hiden level biases
    """
    # Array preparation
    par1_input_coef = parent1.coefs_[0].ravel()
    par1_hidden_coef = parent1.coefs_[1].ravel()
    par1_input_bias = parent1.intercepts_[0].ravel()
    par1_hidden_bias = parent1.intercepts_[1].ravel()
    par2_input_coef = parent2.coefs_[0].ravel()
    par2_hidden_coef = parent2.coefs_[1].ravel()
    par2_input_bias = parent2.intercepts_[0].ravel()
    par2_hidden_bias = parent2.intercepts_[1].ravel()
    # Random crossover point
    # Swap and switch
    # child1 = parent1(end) + parent2(begining)
    # child2 = parent2(end) + parent1(begining)
    # input layer weights
    point = random.choice(range(len(par1_input_coef) - 1)) + 1
    end = len(par1_input_coef)
    child1_input_coef = np.zeros(len(par1_input_coef))
    child1_input_coef[:end - point] = par1_input_coef[point:]
    child1_input_coef[end - point:] = par2_input_coef[:point]
    child2_input_coef = np.zeros(len(par2_input_coef))
    child2_input_coef[:end - point] = par2_input_coef[point:]
    child2_input_coef[end - point:] = par1_input_coef[:point]
    # hidden layer weights
    point = random.choice(range(len(par1_hidden_coef) - 1)) + 1
    end = len(par1_hidden_coef)
    child1_hidden_coef = np.zeros(len(par1_hidden_coef))
    child1_hidden_coef[:end - point] = par1_hidden_coef[point:]
    child1_hidden_coef[end - point:] = par2_hidden_coef[:point]
    child2_hidden_coef = np.zeros(len(par2_hidden_coef))
    child2_hidden_coef[:end - point] = par2_hidden_coef[point:]
    child2_hidden_coef[end - point:] = par1_hidden_coef[:point]
    # input biases
    point = random.choice(range(len(par1_input_bias) - 1)) + 1
    end = len(par1_input_bias)
    child1_input_bias = np.zeros(len(par1_input_bias))
    child1_input_bias[:end - point] = par1_input_bias[point:]
    child1_input_bias[end - point:] = par2_input_bias[:point]
    child2_input_bias = np.zeros(len(par2_input_bias))
    child2_input_bias[:end - point] = par2_input_bias[point:]
    child2_input_bias[end - point:] = par1_input_bias[:point]
    # hidden bias
    point = random.choice([0, 1])
    child1_hidden_bias = (1 - point) * par1_hidden_bias + point * par2_hidden_bias
    child2_hidden_bias = point * par1_hidden_bias + (1 - point) * par2_hidden_bias
    # Return two children as ANNs
    child1_mlp = copy.copy(parent1)  # That will save all attributs of the object MLPClassifier
    child2_mlp = copy.copy(parent2)
    child1_mlp.coefs_ = [child1_input_coef.reshape(4, 3), child1_hidden_coef.reshape(3, 1)]
    child2_mlp.coefs_ = [child2_input_coef.reshape(4, 3), child2_hidden_coef.reshape(3, 1)]
    child1_mlp.intercepts_ = [child1_input_bias.reshape(1, 3), child1_hidden_bias]
    child2_mlp.intercepts_ = [child2_input_bias.reshape(1, 3), child2_hidden_bias]
    return child1_mlp, child2_mlp


def create_gene_space(mlp):
    """"
    Create list from which will be layer and index chosen for mutation
    """
    gene_space = []
    for i in range(len(mlp.coefs_[0].ravel())):
        gene_space.append(['inc' + str(i)])
    for i in range(len(mlp.coefs_[1].ravel())):
        gene_space.append(['hic' + str(i)])
    for i in range(len(mlp.intercepts_[0].ravel())):
        gene_space.append(['inb' + str(i)])
    for i in range(len(mlp.intercepts_[1].ravel())):
        gene_space.append(['hib' + str(i)])
    return np.array(gene_space)


def mutation_param(generation):
    """
    Calculate mean of the weights in the same place in the ANNs in the generation
    """
    next_gen_inc = []
    next_gen_hidc = []
    next_gen_inb = []
    next_gen_hidb = []
    for mlp in generation:
        next_gen_inc.append(mlp.coefs_[0].ravel())
        next_gen_hidc.append(mlp.coefs_[1].ravel())
        next_gen_inb.append(mlp.intercepts_[0].ravel())
        next_gen_hidb.append(mlp.intercepts_[1].ravel())
    input_coef_mean = np.array(next_gen_inc).mean(axis=0)
    hidden_coef_mean = np.array(next_gen_hidc).mean(axis=0)
    input_bias_mean = np.array(next_gen_inb).mean(axis=0)
    hidden_bias_mean = np.array(next_gen_hidb).mean(axis=0)
    return input_coef_mean, hidden_coef_mean, input_bias_mean, hidden_bias_mean


def mutate(next_generation, mutation_rate, inc, hic, inb, hib):
    """
    Mutate children based on mutation rate.
    Mutation will substitute randomly chosen weight with the mean of the
    generaton's weight: inc, hic, inb, hib.
    """
    mutation = np.random.choice([0, 1], size=len(next_generation),
                                p=[1 - mutation_rate, mutation_rate])
    nonzeromutation = np.nonzero(mutation)[0]
    rand_genes = np.random.choice(gene_space, size=sum(mutation))
    #    print(mutation)
    #    print(nonzeromutation)
    #    print(rand_genes)
    gene_idx = 0
    children = next_generation
    for i in nonzeromutation:
        mlp = next_generation[i]
        weight_idx = int(rand_genes[gene_idx][3:])
        layer = rand_genes[gene_idx][:3]
        if layer == 'inc':
            mlp.coefs_[0].ravel()[weight_idx] = inc[weight_idx]
        elif layer == 'hic':
            mlp.coefs_[1].ravel()[weight_idx] = hic[weight_idx]
        elif layer == 'inb':
            mlp.intercepts_[0].ravel()[weight_idx] = inb[weight_idx]
        else:
            mlp.intercepts_[1].ravel()[weight_idx] = hib[weight_idx]
        children[i] = mlp
        gene_idx += 1
    return children


def generate_children(parents, parent_idx):
    """
    Generate childrens from all sorted parent pairs
    """
    next_generation = []
    for i in range(len(parents) // 2):
        child1, child2 = breed_swap_switch(parents[parent_idx[i * 2]], parents[parent_idx[i * 2 + 1]])
        next_generation.extend((child1, child2))
    return next_generation


# ===========visualization==========================
def plotdata(scores, generations):
    # Takes generational scores in a numpy array and plot against generations
    avg_scores = scores.mean(axis=1)
    max_scores = scores.max(axis=1)
    min_scores = scores.min(axis=1)
    # create a figure and plot the maximum scores , average scores and Minimum scores for each generaion
    figsize = [10, 12]
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.set_title('Generation reward plot')
    ax.set_xlabel('Generations')
    ax.set_ylabel('Reward')
    ax.plot(avg_scores, lw=1, label='Average rewards')
    ax.plot(max_scores, label='Maximum rewards')
    ax.plot(min_scores, label='Minimum rewards')
    ax.legend(loc=1, ncol=3)
    # create dynamic ticks and tick labes for the x axis
    step = int(generations / 10)
    tick_array = np.arange(1, generations, step)
    ax.set_xticks(tick_array)
    ax.set_xticklabels(tick_array)
    plt.show()


# ================================================


population_size = 50  # Between 10 and 100, depending on CPU
generations = 10  # Between 10 and 20
mutation_rate = 0.001  # Between 0.1 and 0.001
iterations = np.inf
env = gym.make('CartPole-v1')
# Set the maximum number of steps for each cartpole ballance
env._max_episode_steps = 200  # np.inf , for infinity steps

# inlayer_size = env.observation_space.sample().shape[0] # 4 input nodes
inlayer_size = env.observation_space.sample().shape[0]  # 4

# inlayer_size=2
outlayer_size = 1
hlayer_size = int(2 / 3 * inlayer_size + outlayer_size)

# Generate the initial population
ann_population = initial_population(population_size, hlayer_size)

# Prepare gene space as a starting point for mutation
mlp = ann_population[0]
gene_space = create_gene_space(mlp).ravel()

# Iterate over the generations
pop = 1
best_score = 0
tot_inc = 0
tot_hic = 0
tot_inb = 0
tot_hib = 0
rewards_summary = np.zeros((generations, population_size), dtype=float)
while pop <= generations:
    print('\npopulation ', pop)
    # Calculate the score of each ANN in the population (generation)
    ann_rewards = population_score(ann_population, iterations, time_between=0.1, delay=0.05)
    rewards_summary[pop - 1] = ann_rewards

    # Find two parents based on rewards
    parent_idx, parent1, best, average = find_parents(ann_rewards)
    # The best scored from the population (parent1)
    print('\nThe average score in the population ', pop, ' :', average)
    print('\nThe best score in the population ', pop, ' :', best)
    print('\nThe best scored ANN')
    print('\nWeights\n', parent1.coefs_)
    print('\nBiases\n', parent1.intercepts_)
    if best >= best_score:
        best_score = best
        best_mlp = copy.copy(parent1)
    # Calculate average weights and bias in the generation
    inc, hic, inb, hib = mutation_param(ann_population)

    # Calculate sum of average weights and biases through generations
    tot_inc += inc
    tot_hic += hic
    tot_inb += inb
    tot_hib += hib

    # Breeding 2: From 25 parent pairs formed based on probability create 25 pairs
    # of children (the whole new population)
    # Chromosome = whole ANN
    # Gene = the weight from the ANN
    # Crossover
    # Mutation: to avoid a local optimum

    # Breeding 2
    next_generation = generate_children(ann_population, parent_idx)
    # Mutate children
    ann_population = mutate(next_generation, mutation_rate, inc, hic, inb, hib)
    pop += 1

# Average weights and biases across all generations
avg_inc = tot_inc / generations
avg_hic = tot_hic / generations
avg_inb = tot_inb / generations
avg_hib = tot_hib / generations
avg_mlp = copy.copy(parent1)
avg_mlp.coefs_ = [avg_inc.reshape(4, 3), avg_hic.reshape(3, 1)]
avg_mlp.intercepts_ = [avg_inb.reshape(1, 3), avg_hib]

# The best network across all generations
print('\nThe best network across all generations with the score: ', best_score)
print('\nWeights\n', best_mlp.coefs_)
print('\nBiases\n', best_mlp.intercepts_)
# The average network across all generations
print('\nThe average network across all generations')
print('\nWeights\n', avg_mlp.coefs_)
print('\nBiases\n', avg_mlp.intercepts_)

# time.sleep(5)
# Run the best and the average network
ann_population = []
ann_population.append(best_mlp)
ann_population.append(avg_mlp)
ann_rewards = population_score(ann_population, iterations, time_between=5, delay=0.1)
print('\nbest score: ', ann_rewards[0])
print('\naverage score: ', ann_rewards[1])

plotdata(rewards_summary, generations)
