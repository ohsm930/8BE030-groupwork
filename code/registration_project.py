"""
Project code for image registration topics.
"""

import numpy as np
import matplotlib.pyplot as plt
import registration as reg
from IPython.display import display, clear_output


def intensity_based_registration_demo():

    # read the fixed and moving images
    # change these in order to read different images
    I = plt.imread('../data/image_data/1_1_t1.tif')
    Im = plt.imread('../data/image_data/1_1_t2.tif')

    # initial values for the parameters
    # we start with the identity transformation
    # most likely you will not have to change these
    #x = np.array([0., 0., 0.05]) (exp 1)
    x = np.array([0., 1., 1., 0., 0., 0., -0.1])

    # NOTE: for affine registration you have to initialize
    # more parameters and the scaling parameters should be
    # initialized to 1 instead of 0

    # the similarity function
    # this line of code in essence creates a version of rigid_corr()
    # in which the first two input parameters (fixed and moving image)
    # are fixed and the only remaining parameter is the vector x with the
    # parameters of the transformation
    #fun = lambda x: reg.rigid_corr(I, Im, x, return_transform=False) #(exp 1)
    #fun = lambda x: reg.affine_corr(I, Im, x, return_transform=False) #(exp 2, 3)
    fun = lambda x: reg.affine_mi(I, Im, x, return_transform=False) #(exp 4, 5)

    # the learning rate
    mu = 0.001

    # number of iterations
    num_iter = 50

    iterations = np.arange(1, num_iter+1)
    similarity = np.full((num_iter, 1), np.nan)

    fig = plt.figure(figsize=(14,6))

    # fixed and moving image, and parameters
    ax1 = fig.add_subplot(121)

    # fixed image
    im1 = ax1.imshow(I)
    # moving image4
    im2 = ax1.imshow(I, alpha=0.7)
    # parameters
    txt = ax1.text(0.3, 0.95,
        np.array2string(x, precision=5, floatmode='fixed'),
        bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10},
        transform=ax1.transAxes)

    # 'learning' curve
    ax2 = fig.add_subplot(122, xlim=(0, num_iter), ylim=(0, 1))

    learning_curve, = ax2.plot(iterations, similarity, lw=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Similarity')
    ax2.grid()

    # perform 'num_iter' gradient ascent updates
    for k in np.arange(num_iter):

        # gradient ascent
        g = reg.ngradient(fun, x)
        x += g*mu

        # for visualization of the result
        #S, Im_t, _ = reg.rigid_corr(I, Im, x, return_transform=True) #(exp 1)
        #S, Im_t, _ = reg.affine_corr(I, Im, x, return_transform=True) #(exp 2, 3)
        S, Im_t, _ = reg.affine_mi(I, Im, x, return_transform=True) #(exp 4, 5)

        clear_output(wait = True)

        # update moving image and parameters
        im2.set_data(Im_t)
        txt.set_text(np.array2string(x, precision=5, floatmode='fixed'))

        # update 'learning' curve
        similarity[k] = S
        learning_curve.set_ydata(similarity)

        display(fig)
