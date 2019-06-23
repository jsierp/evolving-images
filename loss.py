import imageio
import numpy as np
from time import time

from scipy.misc import imresize
from utils import construct_image


class SquaredDifferenceLoss(object):

    def __init__(self, target_image_path, internal_shape, k=3, verbose=True):
        self.target_image = imresize(imageio.imread(target_image_path), internal_shape)
        self.verbose = verbose
        self.k = k

    def __call__(self, population):
        population_size = population.shape[0]
        result = np.empty(population_size, dtype=np.float64)
        height, width, depth = self.target_image.shape

        start_timestamp = time()
        for i in range(population_size):
            image = construct_image(population[i], (height, width), self.k)
            result[i] = self.pixelwise_diff(image, self.target_image)
        end_timestamp = time()
        time_elapsed = end_timestamp - start_timestamp
        if self.verbose:
            print('[SquaredDifferenceLoss]: Processing batch of ' + str(population_size) + ' individuals took ' +
                  str(time_elapsed) + ' s.')
        return result

    @staticmethod
    def pixelwise_diff(generated_image, target_image):
        '''
        Returns summed pixelwise difference between generated and target image.
        :param generated_image: shape - (h, w, 4) - generated image in rgba format
        :param target_image: shape - (h, w, 4) - target image in rgba format
        :return: pixelwise difference between given images
        '''

        return np.sum((generated_image.astype(np.int64) - target_image.astype(np.int64)) ** 2)
