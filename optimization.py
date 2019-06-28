#%%
import numpy as np
from time import time
from matplotlib import pyplot as plt
from utils import construct_image
from loss import SquaredDifferenceLoss
from sga import SGA
from es import es
from es2 import es2
from simple import simple

if __name__ == '__main__':
    iterations = 100000
    img_shape = (200, 200)
    polygons = 400
    init_number_of_polygons = 20
    mu = 200
    lambda_ = 50
    plt.imshow(SquaredDifferenceLoss('Mona_Lisa.png', img_shape).target_image)
    plt.show()

    # best_chromosome, obj_min_history, obj_mean_history, obj_max_history = \
    #     es2(mu, lambda_, polygons, init_number_of_polygons, img_shape,
    #         'Mona_Lisa.png', number_of_iterations=iterations, init_load='')

    # best_chromosome, obj_min_history, obj_mean_history, obj_max_history, = \
    #     SGA(100, polygons, 3, img_shape, 'Mona_Lisa.png', 200, number_of_iterations=iterations)

    # d = polygons * 10
    # best_chromosome, obj_min_history, obj_mean_history, obj_max_history, sigmas_history = \
    #     es(500, polygons, 3, img_shape, 'Mona_Lisa.png', 1000, 0.07,
    #        1/np.sqrt(2*d), 1/np.sqrt(2*np.sqrt(d)), number_of_iterations=iterations)

    wdir = 'history/local2/'
    best_chromosome, obj_min_history, obj_mean_history, obj_max_history = \
        simple(mu, lambda_, polygons, init_number_of_polygons, img_shape,
            'Mona_Lisa.png', number_of_iterations=iterations, init_load=True, wdir=wdir)

    best_chromosome.dump(wdir + 'best_chromosome')
    obj_min_history.dump(wdir + 'obj_min_history')
    obj_mean_history.dump(wdir + 'obj_mean_history')
    obj_max_history.dump(wdir + 'obj_max_history')

    plt.plot(np.arange(iterations), obj_min_history, label='min')
    plt.plot(np.arange(iterations), obj_max_history, label='mean')
    plt.plot(np.arange(iterations), obj_mean_history, label='max')
    plt.legend()
    plt.show()
    # plt.plot(np.arange(iterations), sigmas_history, label='sigmas')
    # plt.legend()
    # plt.show()

    plt.imshow(construct_image(best_chromosome, (100, 100), 3))
    plt.show()
