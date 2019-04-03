import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from scipy.stats import rankdata
import matplotlib as plt

env = gym.make('CartPole-v0')
exploration_rate = 1
max_episodes = 1000


def create_model():
    # first create the foundation of the layers using Sequential
    model = Sequential()

    # capture the num of states and actions (returns 2 possible movers for CartPole) for use in defining the layers
    states = env.observation_space.shape[0]
    actions = env.action_space.n

    # Define the layers using Dense (3 layers - 1 input, 1 hidden, 1 output)
    # First add input layer with 4 nodes and then the hidden layer with 3 nodes
    model.add(Dense(3, input_dim=states, activation='relu'))

    # Then add the output layer
    model.add(Dense(actions, activation='linear'))

    # Create the model
    # We choose Adam optimizer with a learning rate of 0.01
    opt = optimizers.Adam(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


def initialize_pop(population):
    gens = []
    for i in range(population):
        model = create_model()
        gens.append(model)
    return gens


def get_individual_result(individual):
    observation = env.reset()
    score = 0
    done = False

    while not done:
        # Generates output predictions for each individual
        action = individual.predict_classes(np.reshape(observation, [1, 4]))
        # print('action ', action[0])
        observation, reward, done, info = env.step(action[0])
        score += reward

    return score


def get_generation_result(generation, pop_size):
    reward = []

    for individual in generation:
        individual_reward = []
        for i in range(pop_size):
            individual_reward = [get_individual_result(individual)]
        reward.append(individual_reward)
    return generation, np.array(reward)


# select parents for further mutation
def select_parents(generation, reward, size):
    # print('rewards are:', reward)
    x = rankdata(np.median(1000 / reward, axis=1))
    y = rankdata(np.sum(1000 / reward, axis=1))
    z = x + y
    parents = z.max() / z
    parents = parents.argsort()[-size::][::-1]
    print('best parent: ', parents)
    print(np.median(1000 / reward, axis=1))
    print(np.sum(1000 / reward, axis=1))

    fittest = [generation[i] for i in parents]
    weights = [[i.get_weights()[0] for i in individual.layers] for individual in fittest]

    return fittest, weights


def mutation(gene_space, discount_rate, mutations):
    discount = discount_rate * max_episodes
    mutate = np.random.randint(mutations)
    if np.random.rand() <= 1:
        for _ in range(mutate):
            x = np.random.randint(0, gene_space.shape[0])

            step = np.random.randint(0, discount) / max_episodes
            operation = np.random.choice([0, 1])
            if operation == 1:
                gene_space[x] += step
            else:
                gene_space[x] -= step

    return gene_space


def reshape_gene(gene_space, weight):
    counter = 0
    new_gene = []

    # print('==========================================')
    # print(weight)
    # print(gene_space)
    # print('----------------------------------')
    for item in weight:
        print('weight', item[0], ' and ', item[1], ' together ', weight, ' , ', item)
        x = item[0] + item[1]
        print('==========================================5')
        print(item)
        print('-----------------------------------------5')
        if item[0]==3 and len(gene_space[counter:(counter + x)])!=0:
            new_gene.append(gene_space[counter:(counter + x)].reshape(4,3))
            # new_gene.append(np.stack(gene_space[counter:(counter + x)]))
        new_gene.append(np.zeros((item[1])))
        counter += x
    return new_gene


def next_population(genes, individual_count, discount_rate, mutations):
    new_genes = list(genes)
    # new_genes = np.array(genes)

    new_weights = []
    new_gen = []

    for i in range(individual_count):
        np.random.shuffle(genes)
        gene, weight = np.copy(genes[0][0]), np.copy(genes[0][1])
        x = (mutation(gene, discount_rate, mutations), weight)
        new_genes.append(x)

    for item in new_genes:
        new_gene_shape = reshape_gene(item[0], item[1])
        new_weights.append(new_gene_shape)
        new_weights.append(np.stack(item).reshape(4,3))
        # ========================================================================================================================================================
    for item in new_weights:
        model = create_model()
        model.set_weights(item)
        new_gen.append(model)
    return new_gen


def combine(fit_weights):
    genes, weights = [], []
    for items in fit_weights:
        output = np.array([])
        w = []
        for j in items:
            w.append(j.shape)
            output = np.append(output, w)
        genes.append(output)
        weights.append(w)
    res = list(zip(genes, weights))

    return res


def plotdata(scores, generations):
    """
    Vizualisation.
    Takes generational scores in a numpy array and plot against generations
    """
    avg_scores = scores.mean(axis=1)
    max_scores = scores.max(axis=1)
    min_scores = scores.min(axis=1)
    # Create a figure and plot the maximum scores, average scores and 
    # minimum scores for each generation
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


def main():
    training_data = []
    pop_size = 50
    generations = 15
    discount_rate = 0.5
    counter = 0
    mutation_parent_num = 2

    rewards_summary = np.zeros((generations, pop_size), dtype=float)

    print('Initial Population : ')
    gen, rew = get_generation_result(generation=initialize_pop(population=pop_size), pop_size=5)
    #
    # print('==========================================1')
    # # print(gen)
    # # print(rew)
    # print('-----------------------------------------1')

    children, weights = select_parents(gen, rew, mutation_parent_num)

    new_genes = combine(weights)
    # print('==========================================2')
    # print(len(new_genes))
    # print('-----------------------------------------2')

    rewards_summary = np.zeros((generations, pop_size), dtype=float)

    for i in range(generations):


        new_gen = next_population(new_genes, pop_size - 2, discount_rate, generations)


        next_gen, next_rew = get_generation_result(new_gen, 10)
        # ==============================
        rewards_summary[i] = next_rew
        # ===============================
        reward = next_rew.shape[0] * next_rew.shape[1] * max_episodes
        reward_sum = reward.sum()
        change = ((reward - reward_sum) / reward)

        best_individuals, best_weights = select_parents(new_gen, next_rew, mutation_parent_num)
        best_genes = combine(best_weights)
        counter += 1

    print('Total generations: ', counter, ' and total cartpole games: ', counter * pop_size)
    plotdata(rewards_summary, generations)

main()