import numpy as np
from time import time
from matplotlib import pyplot as plt
from utils import construct_image
from loss import SquaredDifferenceLoss


def es2(population_size, number_of_offspring, num_of_polygons, init_num_of_polygons, image_shape, image_path,
        k=3, crossover_probability=0.95, mutation_probability=0.95, number_of_iterations=250, init_load=""):

    time0 = time()
    loss = SquaredDifferenceLoss(image_path, image_shape, verbose=False)

    if not init_load:
        current_num_of_polygons = init_num_of_polygons
        current_population = random_population(population_size, current_num_of_polygons, k)
    else:
        current_population = np.load(init_load, allow_pickle=True)
        current_num_of_polygons = current_population.shape[1]

    best_objective_value = np.Inf
    best_chromosome = np.zeros((1, current_num_of_polygons, 10))

    obj_max_history = np.empty(number_of_iterations, dtype=np.float64)
    obj_mean_history = np.empty(number_of_iterations, dtype=np.float64)
    obj_min_history = np.empty(number_of_iterations, dtype=np.float64)

    objective_values = loss(current_population)
    counter = 0

    for t in range(number_of_iterations):
        # selecting the parent indices by the roulette wheel method
        # minimize loss
        fitness_values = objective_values.max() - objective_values
        if fitness_values.sum() > 0:
            fitness_values = fitness_values / fitness_values.sum()
        else:
            fitness_values = np.ones(population_size) / population_size
        parent_indices = np.random.choice(population_size, number_of_offspring, True, fitness_values).astype(np.int64)

        # creating the children population
        children_population = np.zeros((number_of_offspring, current_num_of_polygons, 4 + 2*k))
        for i in range(int(number_of_offspring / 2)):
            if np.random.random() < crossover_probability:
                children_population[2 * i, :], children_population[2 * i + 1, :], \
                    = crossover(
                        current_population[parent_indices[2 * i], :],
                        current_population[parent_indices[2 * i + 1], :])
            else:
                children_population[2 * i, :], children_population[2 * i + 1, :] = \
                    current_population[parent_indices[2 * i], :].copy(), \
                    current_population[parent_indices[2 * i + 1]].copy()
        if np.mod(number_of_offspring, 2) == 1:
            children_population[-1, :] = current_population[parent_indices[-1], :]

        # mutating the children population
        for i in range(number_of_offspring):
            children_population[i] = posnoise_mutation(children_population[i])
            children_population[i] = colornoise_mutation(children_population[i])
            children_population[i] = permute_mutation(children_population[i])
            children_population[i] = reinit_polygons_mutation(children_population[i])

        if counter > 1:
            current_num_of_polygons += 1
            print('Adding a new polygon, now: ', current_num_of_polygons)
            children_population = np.hstack([current_population, random_population(population_size, 1, k)])
            current_population = np.hstack([current_population, random_population(population_size, 1, k)])
            objective_values = loss(current_population)
            counter = 0
            best_objective_value = np.Inf

        children_population = limit(children_population)

        # evaluating the objective function on the children population
        children_objective_values = loss(children_population)

        # replacing the current population by (Mu + Lambda) Replacement
        objective_values = np.hstack([objective_values, children_objective_values])
        current_population = np.vstack([current_population, children_population])

        indices = np.argsort(objective_values)
        current_population = current_population[indices[:population_size], :]
        objective_values = objective_values[indices[:population_size]]

        # recording some statistics
        best_index = objective_values.argmin()
        if objective_values[best_index] < best_objective_value:
            best_objective_value = objective_values[best_index]
            best_chromosome = current_population[best_index]
            counter = 0
        elif current_num_of_polygons < num_of_polygons:
            counter += 1

        obj_min_history[t] = objective_values.min()
        obj_mean_history[t] = objective_values.mean()
        obj_max_history[t] = objective_values.max()

        print('%3d %14.8f %12.8f %12.8f %12.8f' % (
            t, time() - time0, obj_min_history[t], obj_mean_history[t], obj_max_history[t]))

        if (t <= 100 and t % 10 == 0) or (t > 100 and t % 100 == 0):
            plt.imshow(construct_image(best_chromosome, image_shape, k))
            plt.savefig('history/epoch_%d.png' % t)
            plt.show()
            best_chromosome.dump('history/es2_best_chromosome')
            obj_min_history.dump('history/es2_obj_min_history')
            obj_mean_history.dump('history/es2_obj_mean_history')
            obj_max_history.dump('history/es2_obj_max_history')
            current_population.dump('history/es2_400_checkpoint')

    current_population.dump('history/es2_400_final')
    return best_chromosome, obj_min_history, obj_mean_history, obj_max_history


def random_population(population_size, num_of_polygons, k):
    return np.stack([random_individual(num_of_polygons, k)
                     for _ in range(population_size)], axis=0)


def random_individual(num_of_polygons, k):
    individual = np.random.uniform(size=(num_of_polygons, 4+2*k))
    alphas = np.random.uniform(size=num_of_polygons)
    individual[:, 3] = np.maximum(alphas, 0.2)
    return individual


def crossover(ind1, ind2):
    swap = np.random.random(len(ind1)) < 0.5
    swap_not = np.logical_not(swap)
    child1, child2 = np.empty_like(ind1), np.empty_like(ind1)
    child1[swap], child1[swap_not] = ind1[swap], ind2[swap_not]
    child2[swap], child2[swap_not] = ind2[swap], ind1[swap_not]
    return child1, child2


def choose_polygons(n, mi=1.):
    p = float(mi) / n
    to_choose = np.random.binomial(n, p)
    inds = np.random.choice(n, to_choose, replace=False)
    return inds


def posnoise_mutation(individual):
    inds = choose_polygons(len(individual))
    individual[inds, 4:] += 0.07 * np.random.standard_normal(size=(len(inds), 3*2))
    return individual


def colornoise_mutation(individual):
    inds = choose_polygons(len(individual))
    individual[inds, :4] += 0.15 * np.random.standard_normal(size=(len(inds), 4))
    return individual


def permute_mutation(individual):
    inds = choose_polygons(len(individual))
    individual[inds] = np.random.permutation(individual[inds])
    return individual


def reinit_polygons_mutation(individual, k=3):
    inds = choose_polygons(len(individual))
    individual[inds] = random_individual(len(inds), k)
    return individual


def limit(population):
    population = np.minimum(1., np.maximum(0., population))
    return population
