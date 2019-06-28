import numpy as np
from time import time
from matplotlib import pyplot as plt
from utils import construct_image
from loss import SquaredDifferenceLoss


def simple(population_size, number_of_offspring, num_of_polygons, init_num_of_polygons, image_shape, image_path,
        k=3, crossover_probability=0.95, mutation_probability=0.95, number_of_iterations=250, init_load=False, wdir=''):

    time0 = time()
    loss = SquaredDifferenceLoss(image_path, image_shape, verbose=False)

    best_objective_value = np.Inf
    if init_load:
        best_chromosome = np.load(wdir + 'best_chromosome', allow_pickle=True)
        obj_max_history = np.load(wdir + 'obj_max_history', allow_pickle=True)
        obj_mean_history = np.load(wdir + 'obj_mean_history', allow_pickle=True)
        obj_min_history = np.load(wdir + 'obj_min_history', allow_pickle=True)
    else:
        best_chromosome = random_individual(init_num_of_polygons, k)
        obj_max_history = np.empty(number_of_iterations, dtype=np.float64)
        obj_mean_history = np.empty(number_of_iterations, dtype=np.float64)
        obj_min_history = np.empty(number_of_iterations, dtype=np.float64)

    current_num_of_polygons = len(best_chromosome)

    best_objective_value = loss(best_chromosome.reshape((1, best_chromosome.shape[0], best_chromosome.shape[1])))[0]
    rate = 0.02
    success_rate = 1. / number_of_offspring

    for t in range(58000, number_of_iterations):
        # selecting the parent indices by the roulette wheel method
        # minimize loss
        # creating the children population

        # mutating the children population
        if rate < 0.01 and current_num_of_polygons < num_of_polygons:
            children_population = best_chromosome + np.zeros((number_of_offspring * 10, current_num_of_polygons, 2*k+4))
            children_population = np.hstack([children_population, random_population(number_of_offspring * 10, 1, k)])
            best_chromosome = np.vstack([best_chromosome, random_individual(1, k)])
            best_objective_value = loss(best_chromosome.reshape((1, best_chromosome.shape[0], best_chromosome.shape[1])))[0]

            rate = 0.02
            current_num_of_polygons += 1
            print('Adding a new polygon, now: ', current_num_of_polygons)
        else:
            children_population = best_chromosome + np.zeros((number_of_offspring, current_num_of_polygons, 2*k+4))

        for i in range(number_of_offspring):
            children_population[i] = posnoise_mutation(children_population[i], rate)
            children_population[i] = colornoise_mutation(children_population[i], rate)
            children_population[i] = permute_mutation(children_population[i], 20 * rate)

        children_population = limit(children_population)

        # evaluating the objective function on the children population
        children_objective_values = loss(children_population)

        # replacing the current population by (Mu + Lambda) Replacement
        successful = (children_objective_values < best_objective_value).sum()
        successful = min(successful, number_of_offspring)

        success_rate = 0.9 * success_rate + 0.1 * successful / number_of_offspring

        if success_rate < 1. / number_of_offspring:
            rate *= (1 + (success_rate - 1. / number_of_offspring)) ** 4
        else:
            rate *= (1 + (success_rate - 1. / number_of_offspring))

        # recording some statistics
        best_index = children_objective_values.argmin()
        if children_objective_values[best_index] < best_objective_value:
            best_objective_value = children_objective_values[best_index]
            best_chromosome = children_population[best_index]

        obj_min_history[t] = children_objective_values.min()
        obj_mean_history[t] = children_objective_values.mean()
        obj_max_history[t] = children_objective_values.max()

        print('%3d %14.8f %12.8f %12.8f %12.8f SC %12.8f SR %12.8f RA %12.8f' % (
            t, time() - time0, obj_min_history[t], obj_mean_history[t], obj_max_history[t],
            float(successful)/number_of_offspring, success_rate, rate))

        if t == 100 or t < 10000 and t % 200 == 0 or t % 400 == 0:
            plt.imshow(construct_image(best_chromosome, image_shape, k))
            plt.savefig(wdir + 'epoch_%d.png' % t)
            plt.show()
            best_chromosome.dump(wdir + 'best_chromosome')
            obj_min_history.dump(wdir + 'obj_min_history')
            obj_mean_history.dump(wdir + 'obj_mean_history')
            obj_max_history.dump(wdir + 'obj_max_history')

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


def posnoise_mutation(individual, rate):
    ind = np.random.randint(len(individual))
    individual[ind, 4:] += 1 * rate * np.random.standard_normal(3*2)
    return individual


def colornoise_mutation(individual, rate):
    ind = np.random.randint(len(individual))
    individual[ind, :4] += 2 * rate * np.random.standard_normal(4)
    return individual


def permute_mutation(individual, rate):
    if np.random.random() < rate:
        x, y = np.random.choice(len(individual), 2, replace=False)
        individual[[x, y]] = individual[[y, x]]
    return individual


def limit(population):
    population = np.minimum(1., np.maximum(0., population))
    return population
