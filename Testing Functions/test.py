# The testing script to reproduce the results we got while testing.

import os.path

from feature_engineering import *
from implementations import *

# Check if data exists and import it

# Using os.path.join to avoid being OS dependent
train_data_folder_path = os.path.join("data", "train.csv")
test_data_folder_path = os.path.join("data", "test.csv")

if os.path.exists(train_data_folder_path) and os.path.exists(test_data_folder_path):
    print('loading data from /data repository')
    y, X, ids_tr = load_csv_data('data/train.csv')

elif os.path.exists('train.csv') and os.path.exists('test.csv'):
    print('loading data from main repository')
    y, X, ids_tr = load_csv_data('train.csv')

else:
    print('There is no train/test file in the directory try importing it and running again')
    sys.exit()

# Best parameters that we found during our testing
best_lambdas_by_jet = [1e-05, 0.00385, 0.00035, 0.000161]
best_degrees_by_jet = [4, 5, 5, 4]


def get_predictions_by_jet(jet_num, x_tr, y_tr, x_te):
    #     Train model
    print('Training for jet number ' + str(jet_num) + '...')
    w, _ = ridge_regression(y_tr, x_tr, best_lambdas_by_jet[jet_num])

    #     Results
    print('Applying the model for jet number ' + str(jet_num) + '...')
    predictions = x_te.dot(w)
    predictions[predictions >= 0] = 1
    predictions[predictions < 0] = -1

    return predictions


accuracy = 0
for jet_to_test in range(4):
    replace, classifier, divide, squarer, logger, deleter, nothing = features_by_jet(jet_to_test)
    X0 = X[seperate_according_to_PRI_jet_num(X)[jet_to_test], :].copy()
    y0 = y[seperate_according_to_PRI_jet_num(X)[jet_to_test]].copy()
    X0 = feature_cleaning(X0, replace, divide, squarer, logger, classifier, deleter, nothing)

    sinx = np.sin(X0)
    cosx = np.cos(X0)
    ex = np.exp(X0)

    X0 = np.append(X0, sinx, axis=1)
    X0 = np.append(X0, cosx, axis=1)
    X0 = np.append(X0, ex, axis=1)

    X0 = build_poly(X0, best_degrees_by_jet[jet_to_test])
    x_tr, x_te, y_tr, y_te = split_data(X0, y0, 0.2, myseed=1)
    print('accuracy for jet ' + str(jet_to_test) + ': ' +
          str(percentage_of_accuracy(get_predictions_by_jet(jet_to_test, x_tr, y_tr, x_te), y_te)))

    accuracy += percentage_of_accuracy(get_predictions_by_jet(jet_to_test, x_tr, y_tr, x_te), y_te)

accuracy /= 4
print('Average overall accuracy: ' + str(accuracy))
