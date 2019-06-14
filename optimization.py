import numpy as np
from time import time
from matplotlib import pyplot as plt
from utils import construct_image
from loss import SquaredDifferenceLoss


def SGA(population_size, num_of_polygons, k, image_shape, image_path, number_of_offspring,
        crossover_probability=0.95, mutation_probability=0.1, mutation_amount=0.15, number_of_iterations=250):

    time0 = time()
    loss = SquaredDifferenceLoss(image_path, image_shape, verbose=False)
    best_objective_value = np.Inf
    best_chromosome = np.zeros((1, num_of_polygons, 7))
    current_population = random_population(population_size, num_of_polygons, k, image_shape)

    obj_max_history = np.empty(number_of_iterations, dtype=np.float64)
    obj_mean_history = np.empty(number_of_iterations, dtype=np.float64)
    obj_min_history = np.empty(number_of_iterations, dtype=np.float64)

    objective_values = loss(current_population)

    for t in range(number_of_iterations):
        # selecting the parent indices by the roulette wheel method
        fitness_values = objective_values.max() - objective_values
        if fitness_values.sum() > 0:
            fitness_values = fitness_values / fitness_values.sum()
        else:
            fitness_values = np.ones(population_size) / population_size
        parent_indices = np.random.choice(population_size, number_of_offspring, True, fitness_values).astype(np.int64)

        # creating the children population
        children_population = np.zeros((number_of_offspring, num_of_polygons, 4 + 2*k))
        for i in range(int(number_of_offspring / 2)):
            if np.random.random() < crossover_probability:
                children_population[2 * i, :], children_population[2 * i + 1, :] = crossover(
                    current_population[parent_indices[2 * i], :].copy(),
                    current_population[parent_indices[2 * i + 1], :].copy())
            else:
                children_population[2 * i, :], children_population[2 * i + 1, :] = current_population[
                                                                                   parent_indices[2 * i], :].copy(), \
                                                                                   current_population[
                                                                                       parent_indices[2 * i + 1]].copy()
        if np.mod(number_of_offspring, 2) == 1:
            children_population[-1, :] = current_population[parent_indices[-1], :]

        # mutating the children population
        for i in range(number_of_offspring):
            children_population[i, :] = mutation(children_population[i, :], mutation_probability, mutation_amount)

        # evaluating the objective function on the children population
        children_objective_values = loss(children_population)

        # replacing the current population by (Mu + Lambda) Replacement
        objective_values = np.hstack([objective_values, children_objective_values])
        current_population = np.vstack([current_population, children_population])

        I = np.argsort(objective_values)
        current_population = current_population[I[:population_size], :]
        objective_values = objective_values[I[:population_size]]

        # recording some statistics
        if best_objective_value > objective_values[0]:
            best_objective_value = objective_values[0]
            best_chromosome = current_population[0, :]

        obj_min_history[t] = objective_values.min()
        obj_mean_history[t] = objective_values.mean()
        obj_max_history[t] = objective_values.max()

        print('%3d %14.8f %12.8f %12.8f %12.8f %12.8f' % (
            t, time() - time0, obj_min_history[t], obj_mean_history[t], obj_max_history[t],
            objective_values.std()))

        if t % 100 == 0:
            plt.imshow(construct_image(best_chromosome, image_shape, k))
            plt.show()
    return best_chromosome, obj_min_history, obj_mean_history, obj_max_history


def random_population(population_size, num_of_polygons, k, image_shape):
    return np.stack([random_individual(num_of_polygons, k, image_shape)
                     for _ in range(population_size)], axis=0)


def random_individual(num_of_polygons, k, image_shape):
    population = np.random.uniform(size=(num_of_polygons, 4 + 2*k))
    alphas = np.random.uniform(size=(num_of_polygons)) * np.random.uniform(size=(num_of_polygons))
    population[:, 4] = np.maximum(alphas, 0.2)
    return population

def crossover(ind1, ind2):
    original_shape = ind1.shape
    ind1_flatten = ind1.reshape(-1)
    ind2_flatten = ind2.reshape(-1)
    split_idx = np.random.randint(0, ind1_flatten.shape[0])
    ind1_mut_flatten = np.concatenate((ind1_flatten[:split_idx], ind2_flatten[split_idx:]))
    ind2_mut_flatten = np.concatenate((ind1_flatten[split_idx:], ind2_flatten[:split_idx]))
    return ind1_mut_flatten.reshape(original_shape), ind2_mut_flatten.reshape(original_shape)


def mutation(individual, mutation_probability, mutation_amount):
    for polygon in individual:
        for i in range(len(polygon)):
            if np.random.random() < mutation_probability:
                polygon[i] += np.random.uniform(-1., 1.) * mutation_amount
                if polygon[i] > 1:
                    polygon[i] = 1.
                if polygon[i] < 0:
                    polygon[i] = 0.

    return individual


if __name__ ==  '__main__':
    iters = 100000
    loss = SquaredDifferenceLoss('Mona_Lisa.png', (75, 75))
    plt.imshow(loss.target_image)
    plt.show()

    best_chromosome, obj_min_history, obj_mean_history, obj_max_history \
        = SGA(50, 125, 3, (75, 75), 'Mona_Lisa.png', 50,
              mutation_probability=0.01, mutation_amount=0.15, number_of_iterations=iters) # wychodzą same zera, to chyba szuka max a nie min.

    print(best_chromosome)
    print(obj_min_history)
    plt.plot(np.arange(iters), obj_min_history, label='min')
    plt.plot(np.arange(iters), obj_max_history, label='mean')
    plt.plot(np.arange(iters), obj_mean_history, label='max')
    plt.legend()
    plt.show()

    plt.imshow(construct_image(best_chromosome, (75, 75), 3))
    plt.show()

    # population = random_population(50, 125, 3, (75, 75))
    # loss = SquaredDifferenceLoss('Mona_Lisa.png')
    # img = construct_image(population[0], (75, 75), 3)
    # plt.imshow(img)
    # plt.show()
    # print(loss(population))
