import numpy as np
import sys
from proj1_helpers import *


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
       Linear regression using gradient descent

       INPUT:
           y           - Predictions
           tx          - Samples
           initial_w   - Initial weights
           max_iters   - Maximum number of iterations
           gamma       - Step size

       OUTPUT:
           Returns the best weights and the loss
   """

    # Define parameters to store w and loss


    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):

        # compute gradient and loss
        grad, err = compute_gradient(y, tx, w)
        loss = calculate_mse(err)

        # updating w by adding minus the gradient
        w = w - gamma * grad

        # store w and loss
        ws.append(w)
        losses.append(loss)

        if n_iter > 1 and np.abs(losses[-1] - losses[-2]) < 10 ** -6:
            break

        print("Gradient Descent({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))

    return ws[-1], losses[-1]


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
        Linear regression using stochastic gradient descent
        
        INPUT:
            y           - Predictions
            tx          - Samples
            initial_w   - Initial weights
            max_iters   - Maximum number of iterations
            gamma       - Step size
            
        OUTPUT:
            Returns the best weights and the loss
    """

    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            # compute stochastic gradient and loss
            grad, _ = compute_stoch_gradient(y_batch, tx_batch, w)
            loss = compute_loss(y, tx, w)

            # update w by adding minus the gradient
            w = w - gamma * grad

            # store w and loss
            ws.append(w)
            losses.append(loss)

        if n_iter > 1 and np.abs(losses[-1] - losses[-2]) < 10 ** -6:
            break

        print("SGD({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))

    return ws[-1], losses[-1]


def least_squares(y, tx):
    """
        Least square regression using normal equations
        
        INPUT:
            y           - Predictions vector
            tx          - Samples
            
        OUTPUT:
            Returns the best weights w and the loss
            
    """
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w_star = np.linalg.solve(a, b)

    loss = compute_loss(y, tx, w_star)
    return w_star, loss


def ridge_regression(y, tx, lambda_):
    """
        Ridge Regression using normal equations
        
        INPUT:
            y           - Predictions
            tx          - Samples
            lambda_     - Regularization parameter
            
        OUTPUT:
            Returns the best weights and the loss
    """
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_rmse(y, tx, w)

    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
        Logistic Regression using gradient descent
        
        INPUT:
            y           - Predictions
            tx          - Samples
            initial_w   - Initial weights
            max_iters   - Maximum number of iterations
            gamma       - Step size
            
        OUTPUT:
            Returns the best weights and the loss
    """
    losses = []
    tx = np.c_[np.ones((y.shape[0], 1)), tx]
    ws = [initial_w]
    w = initial_w

    for iter in range(max_iters):

        # compute gradient, loss
        loss = compute_loss_logistic(y, tx, w)
        grad = compute_gradient_logistic(y, tx, w)

        # update w by adding minus the gradient
        w = w - gamma * grad

        # store w and loss
        ws.append(w)
        losses.append(loss)

        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))

        # converge criterion
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < 10 ** -8:
            break

    print("loss={l}".format(l=compute_loss_logistic(y, tx, w)))

    return ws[-1], losses[-1]


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
       Regularized Logistic Regression using gradient descent

       INPUT:
           y           - Predictions
           tx          - Samples
           lambda_     - Regularization parameter
           initial_w   - Initial weights
           max_iters   - Maximum number of iterations
           gamma       - Step size

       OUTPUT:
           Returns the best weights and the loss
   """


    losses = []
    tx = np.c_[np.ones((y.shape[0], 1)), tx]
    ws = [initial_w]
    w = initial_w

    for iter in range(max_iters):

        # compute gradient and loss
        loss = compute_loss_logistic(y, tx, w) + lambda_ * np.linalg.norm(w) ** 2
        grad = compute_gradient_logistic(y, tx, w) + 2 * lambda_ * w

        # update w by adding minus the gradient
        w = w - gamma * grad

        # store w and loss
        ws.append(w)
        losses.append(loss)

        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))

        # converge criterion
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < 10 ** -8:
            break

    print("loss={l}".format(l=compute_loss_logistic(y, tx, w)))

    return ws[-1], losses[-1]
