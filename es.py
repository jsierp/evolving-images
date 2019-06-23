import numpy as np
from time import time
from matplotlib import pyplot as plt
from utils import construct_image
from loss import SquaredDifferenceLoss


def es(population_size, num_of_polygons, k, image_shape, image_path, number_of_offspring, sigma, tau, tau_0,
       crossover_probability=0.9, number_of_iterations=250):

    time0 = time()
    loss = SquaredDifferenceLoss(image_path, image_shape, verbose=False)
    best_objective_value = np.Inf
    best_chromosome = np.zeros((1, num_of_polygons, 10))
    current_population = random_population(population_size, num_of_polygons, k)
    current_population_sigmas = sigma * np.ones((population_size, num_of_polygons, 4 + 2*k))

    obj_max_history = np.empty(number_of_iterations, dtype=np.float64)
    obj_mean_history = np.empty(number_of_iterations, dtype=np.float64)
    obj_min_history = np.empty(number_of_iterations, dtype=np.float64)
    sigmas_history = np.empty(number_of_iterations, dtype=np.float64)

    objective_values = loss(current_population)

    for t in range(number_of_iterations):
        # selecting the parent indices by the roulette wheel method
        # minimize loss
        fitness_values = objective_values.max() - objective_values
        if fitness_values.sum() > 0:
            fitness_values = fitness_values ** 2 / (fitness_values ** 2).sum()
        else:
            fitness_values = np.ones(population_size) / population_size
        parent_indices = np.random.choice(population_size, number_of_offspring, True, fitness_values).astype(np.int64)

        # creating the children population
        children_population = np.zeros((number_of_offspring, num_of_polygons, 4 + 2*k))
        children_population_sigmas = np.zeros((number_of_offspring, num_of_polygons, 4 + 2*k))
        for i in range(int(number_of_offspring / 2)):
            if np.random.random() < crossover_probability:
                children_population[2 * i, :], children_population[2 * i + 1, :], \
                    children_population_sigmas[2 * i, :], children_population_sigmas[2 * i + 1, :], \
                    = crossover(
                        current_population[parent_indices[2 * i], :],
                        current_population[parent_indices[2 * i + 1], :],
                        current_population_sigmas[parent_indices[2 * i], :],
                        current_population_sigmas[parent_indices[2 * i + 1], :])
            else:
                children_population[2 * i, :], children_population[2 * i + 1, :] = \
                    current_population[parent_indices[2 * i], :].copy(), \
                    current_population[parent_indices[2 * i + 1]].copy()
                children_population_sigmas[2 * i, :], children_population_sigmas[2 * i + 1, :] = \
                    current_population_sigmas[parent_indices[2 * i], :].copy(), \
                    current_population_sigmas[parent_indices[2 * i + 1]].copy()
        if np.mod(number_of_offspring, 2) == 1:
            children_population[-1, :] = current_population[parent_indices[-1], :]
            children_population_sigmas[-1, :] = current_population_sigmas[parent_indices[-1], :]

        # mutating the children population
        children_population_sigmas = children_population_sigmas * np.exp(
            tau * np.random.randn(number_of_offspring, num_of_polygons, 4+2*k) + tau_0 * np.random.randn(number_of_offspring, 1, 1))
        children_population = children_population + children_population_sigmas * np.random.randn(
            number_of_offspring, num_of_polygons, 4+2*k)
        children_population = permute_mutation(children_population)

        children_population = limit(children_population)

        # evaluating the objective function on the children population
        children_objective_values = loss(children_population)

        # replacing the current population by (Mu + Lambda) Replacement
        objective_values = np.hstack([objective_values, children_objective_values])
        current_population = np.vstack([current_population, children_population])
        current_population_sigmas = np.vstack([current_population_sigmas, children_population_sigmas])

        indices = np.argsort(objective_values)
        current_population = current_population[indices[:population_size], :]
        current_population_sigmas = current_population_sigmas[indices[:population_size], :]
        objective_values = objective_values[indices[:population_size]]

        # recording some statistics
        if best_objective_value > objective_values[0]:
            best_objective_value = objective_values[0]
            best_chromosome = current_population[0, :]

        obj_min_history[t] = objective_values.min()
        obj_mean_history[t] = objective_values.mean()
        obj_max_history[t] = objective_values.max()
        sigmas_history[t] = current_population_sigmas.sum()

        print('%3d %14.8f %12.8f %12.8f %12.8f %12.8f %12.8f' % (
            t, time() - time0, obj_min_history[t], obj_mean_history[t], obj_max_history[t],
            objective_values.std(), current_population_sigmas.sum()))
    return best_chromosome, obj_min_history, obj_mean_history, obj_max_history, sigmas_history


def random_population(population_size, num_of_polygons, k):
    return np.stack([random_individual(num_of_polygons, k)
                     for _ in range(population_size)], axis=0)


def random_individual(num_of_polygons, k):
    individual = np.empty((num_of_polygons, 4+2*k))
    individual[:, 4:] = np.random.uniform(size=(num_of_polygons, 2*k))
    alphas = np.random.uniform(size=num_of_polygons)
    individual[:, 0] = np.random.uniform(size=num_of_polygons)
    individual[:, 1] = np.random.uniform(size=num_of_polygons)
    individual[:, 2] = np.random.uniform(size=num_of_polygons)
    individual[:, 3] = np.maximum(alphas, 0.2)
    return individual


def crossover(ind1, ind2, ind1sig, ind2sig):
    swap = np.random.random(len(ind1)) < 0.5
    swap_not = np.logical_not(swap)
    child1, child2 = np.empty_like(ind1), np.empty_like(ind1)
    child1sig, child2sig = np.empty_like(ind1sig), np.empty_like(ind1sig)
    child1[swap], child1[swap_not] = ind1[swap], ind2[swap_not]
    child2[swap], child2[swap_not] = ind2[swap], ind1[swap_not]
    child1sig[swap], child1sig[swap_not] = ind1sig[swap], ind2sig[swap_not]
    child2sig[swap], child2sig[swap_not] = ind2sig[swap], ind1sig[swap_not]
    return child1, child2, child1sig, child2sig


def permute_mutation(population):
    high = population.shape[1] // 10
    for i in range(len(population)):
        to_permute = np.random.randint(0, high=high)
        p = np.random.choice(population.shape[1], to_permute, replace=False)
        population[i, p] = population[i, np.random.permutation(p)]
    return population


def limit(population):
    population = np.minimum(1., np.maximum(0., population))
    return population
