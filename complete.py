
import gym
import numpy as np
import random
from sklearn.neural_network import MLPClassifier
from gym import wrappers
import matplotlib.pyplot as plt

"""
Global parameters
"""
population_size = 50  # Between 10 and 100, depending on CPU
generations = 10  # Between 10 and 20
mutation_rate = 0.1  # Between 0.1 and 0.001
max_iteration = 500 #np.inf
env = gym.make('CartPole-v0')
# env._max_episode_steps=max_iteration

"""
The main function responsible for running the program
"""

def mainF():
    # create initial population
    population = initial_population(population_size)
    rewards_summary = np.zeros((generations, population_size), dtype=float)

    generation_best_anns=[]
    for gen in range(generations):
        print('********************************************************')
        print('Generation', gen + 1)
        print('********************************************************')
        # Calculate the reward score of each ANN in the population
        rewards = population_score(population)
        rewards_summary[gen - 1] = rewards
        # Get Parents
        parents=get_parents(population,rewards)
        generation_best_anns.append(parents[0])
        # make children
        children=make_children(parents)
        # create new population out of children to continue evolution
        population=children
        # break
    plotdata(rewards_summary)
    overal_best=get_best_ann(generation_best_anns,rewards_summary)
    # render_with_ann(overal_best)

def get_best_ann(anns,rewards):
   i= np.argmax(rewards)
   print(i)

"""
    Create an initial population of ANN classifiers
"""
def initial_population(population_size):

    inlayer_size = env.observation_space.sample().shape[0]  # 4 input nodes
    outlayer_size = 1
    hlayer_size = int(2 / 3 * inlayer_size + outlayer_size)

    ann_population = []  # List for ANN population
    for ann in range(population_size):
        mlp = MLPClassifier(batch_size=1, max_iter=1, solver='sgd',
                            activation='relu', learning_rate='invscaling',
                            hidden_layer_sizes=hlayer_size, random_state=1)

        mlp.partial_fit(np.array([env.observation_space.sample()], dtype='int64'),
                        np.array([env.action_space.sample()], dtype='int64'),
                        classes=np.arange(env.action_space.n))

        coef_init =np.array([[0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5],
                             [0.8, 0.8, 0.8],
                             [0.8, 0.8, 0.8]])

        coef_hidden =np.array([ [0.5],[0.5],[0.5]])
        mlp.coefs_ = [coef_init, coef_hidden]

        ann_population.append(mlp)
    return ann_population

"""
Uses each ann to balance the pole as it is partialy fitting the ann returns the score of each of the anns
parameters: ann_population = a list of anns befor training
returns : a list of scores coresponding to each trained ann
"""

def population_score(ann_population):
    """
    Calculate the score of each ANN in the population (generation)
    """
    ann_rewards = np.empty(len(ann_population),dtype=int)
    for i, mlp in enumerate(ann_population):
        total_reward = 0
        env.reset()
        action = random.sample(set(np.arange(env.action_space.n)), 1)[0]
        # action=1
        observation, reward, done, info = env.step(action)  # Initial random action
        # track the action sequence of each of the anns and use it for sanity check if at all
        action_sequence = np.zeros(max_iteration, dtype=int)
        iteration=0
        # weights = np.random.uniform(0.0, 1.0, 4)
        while not done:
            total_reward += reward
            # np.put(action_sequence, [iteration], [action])
            action_sequence[iteration]=action
            action = mlp.predict(observation.reshape(1, -1))[0]  # Reshaped data due
            # action = 1 if np.dot(observation, weights) > 0 else 0
            iteration+=1
            if all(i == action for i in action_sequence[iteration - 12:iteration]) and iteration >= 12: # sanity check
                action_flip = 1 if action == 0 else 0
                action = action_flip
                # print('repeat')
            observation, reward, done, info = env.step(action)
            mlp.partial_fit(observation.reshape(1, -1), np.array([action]))
            # if iteration==5:
            #     break
        ann_rewards[i] = total_reward
        # print(action_sequence)
        # break
    return ann_rewards


"""
    Get the best two parents from an array of parent anns
    paramentes: population = list of anns
              : rewards = corresponding list of rwards
    returns 2 parents.
"""
def get_parents(population,rewards):
#   calculate parent probability from rewards
    ann_prob = rewards / sum(rewards)
#   sort the array indices in ascending order of probal=bilities and get the lasts two indices
    parents_idx = np.argsort(ann_prob)
    parent1 = population[parents_idx[-1]]
    parent2 = population[parents_idx[-2]]
    return parent1, parent2

"""
Generate a list of child anns
input parameters : parents tuple (parent1,parent2)
returns : A list of children anns

"""
def make_children(parents):
    parent1,parent2=parents
    children=[]
    for i in range(int(population_size/2)):
        #get the coefficients and biases of the parrents into flat np arrays
        parent1_input_coef =np.array(parent1.coefs_[0].ravel())
        parent1_hidden_coef =np.array( parent1.coefs_[1].ravel())
        parent1_input_bias = np.array(parent1.intercepts_[0].ravel())
        parent1_hidden_bias =np.array(parent1.intercepts_[1].ravel())
        parent2_input_coef = np.array(parent2.coefs_[0].ravel())
        parent2_hidden_coef =np.array(parent2.coefs_[1].ravel())
        parent2_input_bias = np.array(parent2.intercepts_[0].ravel())
        parent2_hidden_bias = np.array(parent2.intercepts_[1].ravel())

        #perform crossover and optionaly mutation on the array pairs each from a parent
        child_input_coef1,child_input_coef2=crossover(parent1_input_coef,parent2_input_coef)
        child_input_bias1,child_input_bias2=crossover(parent1_input_bias,parent2_input_bias)
        child_hidden_coef1,child_hidden_coef2=crossover(parent1_hidden_coef,parent2_hidden_coef)
        child_hidden_bias1,child_hidden_bias2=crossover(parent1_hidden_bias,parent2_hidden_bias)

        #create two children from the two parents templates with modified genes and coefficients
        child1=parent1
        child1.coefs_ = [child_input_coef1.reshape(4, 3), child_hidden_coef1.reshape(3, 1)]
        child1.intercepts_ = [child_input_bias1.reshape(1, 3), child_hidden_bias1]
        child2=parent2
        child2.coefs=[child_input_coef2.reshape(4, 3), child_hidden_coef2.reshape(3, 1)]
        child2.intercepts_= [child_input_bias2.reshape(1, 3), child_hidden_bias2]

        #add to the list of children
        children.append(child1)
        children.append(child2)

    return children

"""
Perform crossover on a pair of arrays of parents genes randomly
input parameter: parent 1 and corresponding parent2 arrays
returns : array sequences after crossover
"""


def crossover(array1,array2):

   #make a copy of the input arrays
   arr1_toswap=np.copy(array1)
   arr2_toswap=np.copy(array2)
   # select no of swap indices randomly, then randomly create indices corresponding to the no within array indices range
   if len(array1)!=1:
       swap_indices= np.random.randint(0,len(array1)-1,np.random.randint(0,len(array1)))
       # swap the generated indices of the array between the two arrays
       arr1_toswap[swap_indices]=array2[swap_indices]
       arr2_toswap[swap_indices]=array1[swap_indices]

       #mutate where
       mutate(arr1_toswap)
       mutate(arr2_toswap)

   return arr1_toswap, arr2_toswap

"""
For mutating an array on probability based on mutation rate
"""
def mutate(array):
    for i in range(100):
#choose weather to mutate or not based on the mutation rate. mutate the array in place
       if np.random.choice(a=[True,False],replace=True,p=[mutation_rate,1-mutation_rate]):
           mutate_Index=np.random.choice(a=len(array)-1,size=1,replace=True)
           array[mutate_Index]=np.mean(array)

"""
Plots the scores against generations
inut parameter : scores : a numpy array of scores corresponding to each ann in each generation

"""
def plotdata(scores):

    # Takes generational scores in a numpy array and plot against generations
    avg_scores = scores.mean(axis=1)
    max_scores = scores.max(axis=1)
    min_scores = scores.min(axis=1)
    # print(avg_scores)

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

"""
For rendering the cartpole with a chosen ann,such as the best
input parameter:ann  
"""
def render_with_ann(ann):
    #reset the max episode to infinity to allow the pole to ballance for as long as the ann can controll it
    env._max_episode_steps = np.inf
    #specify where to save the mov file of the ballancing pole
    env=wrappers.Monitor(env,'MoveFiles_better',force=True)
    env.reset()
    while not done:
        action = ann.predict(observation.reshape(1, -1))[0]  # Reshaped data due
        observation, reward, done, info = env.step(action)
        env.render()





def average_length():
    print('.....Getting Average lengths')
    weights = np.random.uniform(0.0, 1.0, 4)
    lenths=[]
    count = 0
    done = False
    observation = env.reset()
    while not done:
        count += 1
        action = 1 if np.dot(observation, weights) > 0 else 0
        observation, reward, done, _ = env.step(action)
    print(count)
    return 0

mainF()
# crossover('','')
# mutate()

# average_length()