import numpy as np
import cairo
from matplotlib import pyplot as plt


def construct_image(polygons, image_shape, k):
    '''
    Constructs image from circles description.
    :param circles: shape - (population_size, 4 + 2*k) - Description of circles. Each row contains
    r, g, b, a, x1, y1, ..., xk, yk where:
    :param image_shape - (height, width) - shape of constructed image
    :return: image of shape image_shape generated with given circles
    '''

    height, width = image_shape
    data = np.zeros((height, width, 4), dtype=np.uint8)
    surface = cairo.ImageSurface.create_for_data(data, cairo.FORMAT_ARGB32, height, width)
    cr = cairo.Context(surface)
    #cr.paint()
    for polygon in polygons:
        cr.set_source_rgba(polygon[0], polygon[1], polygon[2], polygon[3])
        cr.move_to(polygon[4] * width, polygon[5] * height)
        for j in range(1, k):
            cr.line_to(polygon[4 + 2 * j] * width, polygon[4 + 2 * j + 1] * height)
        cr.fill()

    return data


if __name__ == '__main__':
    polygons = np.array([[1.0, 0., 0., 1., 1.30, 1.30, 0.30, 0.50, 0.50, 0.50],
                         [0., 0., 1., 0.3, 0.10, 0.10, 1.120, 0.10, 0.100, 1.150],
                         [0.0, 1.0, 0., 0.5, 0.40, 1.40, -.80, 0.50, 0.90, 0.100],])
    img = construct_image(polygons, (200, 200), 3)
    plt.imshow(img)
    plt.show()
