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


def initial_population(population_size, hlayer_size):
    """
    Create an initial population of ANN classifiers
    """
    ann_population = []  # List for ANN population
    for ann in range(population_size):
        mlp = MLPClassifier(batch_size=1, max_iter=1, solver='sgd',
                            activation='relu', learning_rate='invscaling',
                            hidden_layer_sizes=hlayer_size, random_state=1)
        mlp.partial_fit(np.array([env.observation_space.sample()], dtype='int64'),
                        np.array([env.action_space.sample()], dtype='int64'),
                        classes=np.arange(env.action_space.n))
        ann_population.append(mlp)
    return ann_population


def population_score(ann_population, iterations, time_between, delay, render):
    """
    Calculate the score of each ANN in the population (generation)
    """
    i = 0
    ann_rewards = np.zeros(len(ann_population))
    training_population = copy.copy(ann_population)
    for mlp in training_population:
        iteration = 0
        total_reward = 0
        env.reset()
        # if render:
        #     env.render()
            # time.sleep(time_between)
        action = random.sample(set(np.arange(env.action_space.n)), 1)[0]
        same_action = [action, 1]  # Sanity check
        observation, reward, done, info = env.step(action)  # Initial random action
        # if render:
            # env.render()
            # time.sleep(delay)
        while not done and iteration < iterations:
            iteration += 1
            total_reward += reward
            action = mlp.predict(observation.reshape(1, -1))[0]  # Reshaped data due
            # to 2D array expected
            same_action[1] += (action == same_action[0])
            same_action[0] = action
            if same_action[1] == 5:  # Five consecutive same actions
                action = int(not bool(action))  # Possible actions are only 0 an 1.
                # Take the opposite action
                same_action = [action, 1]
            observation, reward, done, info = env.step(action)
            # if render:
                # env.render()
                # time.sleep(delay)
            mlp.partial_fit(observation.reshape(1, -1), np.array([action]))
        ann_rewards[i] = total_reward
        i += 1
    return ann_rewards


def find_parents(ann_rewards, mating_pool):
    """
    Calculate the probabilities of ANNs and find parents ordering.
    Find best and average score.
    """
    prob_idx = 0  # to stop throwing exception, if mating_pool != 2
    best_score = max(ann_rewards)
    average_score = ann_rewards.mean()
    ann_prob = ann_rewards / sum(ann_rewards)
    parents_idx = np.argsort(ann_prob)
    if mating_pool == 1:
        parent1 = ann_population[parents_idx[-1]]
        parent2 = ann_population[parents_idx[-2]]
    if mating_pool == 2:
        prob_idx = np.random.choice(len(ann_rewards), len(ann_rewards), p=ann_prob)
        parent1 = ann_population[parents_idx[-1]]
        parent2 = ann_population[parents_idx[-2]]
    if mating_pool == 3:
        number_in_subgroup = np.random.randint(5, len(ann_rewards), size=1)
        subgroup_idx = np.random.choice(len(ann_rewards), number_in_subgroup, replace=False)
        subgroup_parents_idx = np.argsort(ann_rewards[subgroup_idx])
        print("individuals_in_subgroup: ", number_in_subgroup)
        print("best rewards: ", ann_rewards[subgroup_idx][subgroup_parents_idx[-1]])
        parent1 = [ann_population[i] for i in subgroup_idx][subgroup_parents_idx[-1]]
        parent2 = [ann_population[i] for i in subgroup_idx][subgroup_parents_idx[-2]]
    return parents_idx, prob_idx, parent1, parent2, best_score, average_score


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
    Generate childrens from parents
    """
    next_generation = []
    for i in range(len(parents) // 2):
        child1, child2 = breed_swap_switch(parents[parent_idx[i * 2]], parents[parent_idx[i * 2 + 1]])
        next_generation.extend((child1, child2))
    return next_generation


def plottrial(rewards_stat, generations):
    """
    Vizualisation.
    Takes generational scores in a numpy array and plot against generations
    """
    avg_reward = rewards_stat.mean(axis=2).mean(axis=0)
    max_reward = rewards_stat.max(axis=2).mean(axis=0)
    min_reward = rewards_stat.min(axis=2).mean(axis=0)
    # Create a figure and plot the maximum scores, average scores and 
    # minimum scores for each generation
    figsize = [10, 12]
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.set_title('Generation reward plot')
    ax.set_xlabel('Generations')
    ax.set_ylabel('Reward')
    ax.plot(avg_reward, lw=1, label='Average rewards')
    ax.plot(max_reward, label='Maximum rewards')
    ax.plot(min_reward, label='Minimum rewards')
    ax.legend(loc=1, ncol=3)
    # create dynamic ticks and tick labes for the x axis
    step = int(generations / 10)
    tick_array = np.arange(1, generations, step)
    ax.set_xticks(tick_array)
    ax.set_xticklabels(tick_array)
    plt.show()


# Initial setting
population_size = 50  # Between 10 and 100, depending on CPU
generations = 15  # Between 10 and 20
mutation_rate = 0.001  # Between 0.1 and 0.001
iterations = np.inf

# mating pool :
# 1- the best ANNs based on their cumulative reward values
# 2- choosing individuals randomly with the probability of being chosen based on fitness level
# 3- random sub-group can be selected from the population and then the individuals with the 
#    highest fitness are selected for the mating pool
mating_pool = 3

# breeding type :
# 1- from 2 best parents
# 2- from 25 different pairs
breeding_type = 1

env = gym.make('CartPole-v1')
# Set the maximum number of steps for each cartpole ballance
env._max_episode_steps = 1000  # np.inf , for infinity steps

inlayer_size = env.observation_space.sample().shape[0]  # 4 input nodes
outlayer_size = 1
hlayer_size = int(2 / 3 * inlayer_size + outlayer_size)

# Generate the initial population
ann_population = initial_population(population_size, hlayer_size)

# Prepare gene space as a starting point for mutation 
mlp = ann_population[0]
gene_space = create_gene_space(mlp).ravel()

# Repeat evolution over the generations trials times
trials = 1000
trial = 0
running_time = np.zeros(trials)
rewards_stat = np.zeros((trials, generations, population_size), dtype=float)
best_score = 0
tot_inc = np.zeros((trials, 12))
tot_hic = np.zeros((trials, 3))
tot_inb = np.zeros((trials, 3))
tot_hib = np.zeros((trials, 1))
for trial in range(trials):
    print('\ntrial ', trial)
    # Generate the initial population
    ann_population = initial_population(population_size, hlayer_size)
    # Iterate over the generations
    start = time.time()  # Start running time measurement
    for pop in range(generations):
        print('\npopulation ', pop)
        # Calculate the score of each ANN in the population (generation)
        ann_rewards = population_score(ann_population, iterations, time_between=0.1, delay=0.05, render=False)
        rewards_stat[trial, pop] = ann_rewards  # Visualization data
        # Find parents based on rewards and/or randomness
        parent_idx, prob_idx, parent1, parent2, best, average = find_parents(ann_rewards, mating_pool)
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
        tot_inc[trial] += inc
        tot_hic[trial] += hic
        tot_inb[trial] += inb
        tot_hib[trial] += hib

        # Mating and breeding 
        # Chromosome = whole ANN  
        # Gene = the weight from the ANN
        # Crossover
        next_generation = []
        if breeding_type == 1 or mating_pool == 3:
            for i in range(population_size // 2):
                child1, child2 = breed_swap_switch(parent1, parent2)
                next_generation.extend((child1, child2))
        elif breeding_type == 2:
            if mating_pool == 1:
                next_generation = generate_children(ann_population, parent_idx)
            elif mating_pool == 2:
                next_generation = generate_children(ann_population, prob_idx)

        # Mutate children : to avoid a local optimum 
        ann_population = mutate(next_generation, mutation_rate, inc, hic, inb, hib)
    end = time.time()  # End running time measurement
    running_time[trial] = end - start

# Average weights and biases across all generations and all trials
avg_inc = tot_inc.mean(axis=0) / generations
avg_hic = tot_hic.mean(axis=0) / generations
avg_inb = tot_inb.mean(axis=0) / generations
avg_hib = tot_hib.mean(axis=0) / generations
avg_mlp = copy.copy(parent1)
avg_mlp.coefs_ = [avg_inc.reshape(4, 3), avg_hic.reshape(3, 1)]
avg_mlp.intercepts_ = [avg_inb.reshape(1, 3), avg_hib]

# The best network across all generations
print('\nmating and breeding strategy: ', mating_pool, '-', breeding_type)
print('\nnumber of trials: ', trials)
print('\nThe best network across all generations with the score: ', best_score)
print('\nWeights\n', best_mlp.coefs_)
print('\nBiases\n', best_mlp.intercepts_)
# The average network across all generations
print('\nThe average network across all generations')
print('\nWeights\n', avg_mlp.coefs_)
print('\nBiases\n', avg_mlp.intercepts_)

# Visualization
plottrial(rewards_stat, generations)
print('\naverage running time: ', running_time.mean())
print('\ntotal running time: ', running_time.sum())

# time.sleep(5)
# Run the best and the average network
env._max_episode_steps = np.inf
ann_population = []
ann_population.append(best_mlp)
ann_population.append(avg_mlp)
ann_rewards = population_score(ann_population, iterations, time_between=5, delay=0.1, render=True)
print('\nbest score: ', ann_rewards[0])
print('\naverage score: ', ann_rewards[1])
