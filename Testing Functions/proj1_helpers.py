"""

Some helper functions for project 1

"""

import csv
import numpy as np

def calculate_mse(e):
    """
       Computes the mean square error given the error vector

       INPUT:
           e           - Error vector

       OUTPUT:
           Returns the mean square vector
    """
    return 1 / 2 * np.mean(e ** 2)


def compute_loss(y, tx, w):
    """
       Computes the mean square error

       INPUT:
           y           - Predictions vector
           tx          - Samples
           w           - Weights

       OUTPUT:
           Returns the mean square vector
    """

    e = y - tx.dot(w)
    return calculate_mse(e)


def compute_rmse(y, tx, w):
    """
       Computes the root-mean-square error

       INPUT:
           y           - Predictions vector
           tx          - Samples
           w           - Weights

       OUTPUT:
           Returns root-mean-square error
    """

    e = y - tx.dot(w)
    return np.sqrt(2 * calculate_mse(e))


def compute_gradient(y, tx, w):
    """
        Computes the gradient for the gradient descent method
        
        INPUT:
            y           - Predictions vector
            tx          - Samples
            w           - Weights
            
        OUTPUT:
            Returns the gradient and the error vector
    """
    e = y - tx.dot(w)
    gradient = (-1 / len(e)) * tx.T.dot(e)

    return gradient, e


def compute_stoch_gradient(y, tx, w):
    """
        Computes the gradient for the stochastic gradient descent method
        
        INPUT:
            y           - Predictions vector
            tx          - Samples
            w           - Weights
            
        OUTPUT:
            Returns the gradient and the error vector
    """
    gradient, e = compute_gradient(y, tx, w)

    return gradient, e


def sigmoid(t):
    """sigmoid function used in logistic regression"""

    return np.exp(t) / (1 + np.exp(-t))


def compute_loss_logistic(y, tx, w):
    """
        Computes the loss for logistic regression, the minus of the log likelihood
        
        INPUT:
            y           - Predictions vector
            tx          - Samples
            w           - Weights
            
        OUTPUT:
            Returns the loss for logistic regression
    """
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))

    return np.squeeze(- loss)


def compute_gradient_logistic(y, tx, w):
    """
        Computes the gradient for logistic regression
        
        INPUT:
            y           - Predictions vector
            tx          - Samples
            w           - Weights
            
        OUTPUT:
            Returns the gradient for logistic regression
    """
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y)

    return grad


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == 'b')] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1

    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id': int(r1), 'Prediction': int(r2)})


def split_data(x, y, ratio, myseed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(myseed)
    # generate random indices
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]
    # create split
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]
    return x_tr, x_te, y_tr, y_te


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree + 1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""
    # get k'th subgroup in test, others in train
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    y_te = y[te_indice]
    y_tr = y[tr_indice]
    x_te = x[te_indice]
    x_tr = x[tr_indice]
    # form data with polynomial degree
    tx_tr = build_poly(x_tr, degree)
    tx_te = build_poly(x_te, degree)

    # ridge regression
    w, loss_tr = ridge_regression(y_tr, tx_tr, lambda_)
    # calculate the loss for train and test data
    loss_te = compute_rmse(y_te, tx_te, w)
    return loss_tr, loss_te, w


def sample_data(y, x, seed, size_samples):
    """sample from dataset."""
    np.random.seed(seed)
    num_observations = y.shape[0]
    random_permuted_indices = np.random.permutation(num_observations)
    y = y[random_permuted_indices]
    x = x[random_permuted_indices]
    return y[:size_samples], x[:size_samples]


def percentage_of_accuracy(y_guessed, y_te):
    """This method returns the percentage of correctness after prediction"""
    S = 0
    for i in range(len(y_guessed)):
        if (y_guessed[i] == y_te[i]):
            S = S + 1
    return 100 * S / len(y_guessed)


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
