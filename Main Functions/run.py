# The main script to reproduce the results we got on Kaggle.

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
    w, _ = ridge_regression(y_tr, x_tr, best_lambdas_by_jet[jet_num])

    #     Results
    predictions = x_te.dot(w)
    predictions[predictions >= 0] = 1
    predictions[predictions < 0] = -1

    return predictions


# For Kaggle submission:

# Training phase

ws = []  # ws for each jet

for jet_to_test in range(4):
    # Get the columns to execute each operation on
    replace, classifier, divide, squarer, logger, deleter, nothing = features_by_jet(jet_to_test)

    # Gets the rows that have a certain jet number from 0 to 3
    X0 = X[seperate_according_to_PRI_jet_num(X)[jet_to_test], :].copy()

    # Gets the values of y where X have a certain jet number from 0 to 3
    y0 = y[seperate_according_to_PRI_jet_num(X)[jet_to_test]].copy()

    # Cleans the matrix by applying all the operations
    X0 = feature_cleaning(X0, replace, divide, squarer, logger, classifier, deleter, nothing)

    # Feature augmentation
    sinx = np.sin(X0)
    cosx = np.cos(X0)
    ex = np.exp(X0)

    X0 = np.append(X0, sinx, axis=1)
    X0 = np.append(X0, cosx, axis=1)
    X0 = np.append(X0, ex, axis=1)

    X0 = build_poly(X0, best_degrees_by_jet[jet_to_test])

    # Train the model using ridge regression
    # and get the w corresponding to a certain jet number

    w, _ = ridge_regression(y0, X0, best_lambdas_by_jet[jet_to_test])

    # Appends the w to the list of ws
    ws.append(w)

# Apply our model on the test file
# load the test file
y, x, ids = load_csv_data('data/test.csv')

# Go through all the values of feature 22
for jet_to_test in range(4):
    # Get the columns to execute each operation on
    replace, classifier, divide, squarer, logger, deleter, nothing = features_by_jet(jet_to_test)

    # Gets the rows that have a certain jet number from 0 to 3
    x0 = x[seperate_according_to_PRI_jet_num(x)[jet_to_test], :].copy()

    # Cleans the matrix by applying all the operations
    x0 = feature_cleaning(x0, replace, divide, squarer, logger, classifier, deleter, nothing)

    # Feature augmentation
    sinx = np.sin(x0)
    cosx = np.cos(x0)
    ex = np.exp(x0)

    x0 = np.append(x0, sinx, axis=1)
    x0 = np.append(x0, cosx, axis=1)
    x0 = np.append(x0, ex, axis=1)

    x0 = build_poly(x0, best_degrees_by_jet[jet_to_test])

    # Apply the model on the augmented feature matrix
    predictions = x0.dot(ws[jet_to_test])

    # Classify our predictions
    predictions[predictions >= 0] = 1
    predictions[predictions < 0] = -1

    # Replace the default values in y by our predictions
    y[seperate_according_to_PRI_jet_num(x)[jet_to_test]] = predictions

# Write our predictions to a csv file
create_csv_submission(ids, y, 'predictions.csv')

print('Operation done successfully. Results are in the file "predictions.csv"')
