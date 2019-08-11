import numpy as np

def seperate_according_to_PRI_jet_num(X):
    ''' 
    Finds the indices of rows that have jet 0,1,2,3 and returns it in an array of 4 arrays
    :param X: The numpy Matrix
    
    :return: the indices of rows that have jet 0,1,2,3 in an array of 4 arrays
    :rtype: array of 4 arrays
    '''
    
    rows = [[], [], [], []]
    for ind, item in enumerate(X):
        rows[int(item[22])].append(ind)
    return rows


def features_by_jet (jet_num):
    ''' 
    Gets all the columns that we need to do the operations on

    :param jet_num: The jet we divide according to

    :return: The columns we need to do operations on
    :rtype: numpy array of arrays
    '''
    
    
#     Case where jet_num = 0
    if (jet_num == 0):
        replace = [0]
        classifier = [[11,-1, 1]]
        divide = [14,15,17,18,20]
        squarer = [1 ,10 ,19,21]
        logger = [0,2 ,3 ,9,13,16]
        deleter = [4,5,6,8,12,22,23,24,25,26,27,28,29]
        nothing = [7]
        
#     Case where jet_num = 1
    elif (jet_num == 1):
        replace = [0]
        classifier = [[11,-1, 1]]
        divide = [14,15,17,18,20,25]
        squarer = [1,3,19]
        logger = [0,2,8,10,13,16,21]
        deleter = [4,5,6,12,26,27,28,9,23,29,22]
        nothing = [7,24]
        
#     Case where jet_num = 2
    elif (jet_num == 2):
        replace = [0]
        classifier = [[11,-1, 1],[12,0,1]]
        divide = [14,15,17,18,20,25,28]
        squarer = [1,3,4,5,9,19]
        logger = [0,2,8,10,13,16,26]
        deleter = [6,21,22,23,29]
        nothing = [7,24,27]

#     Case where jet_num = 3
    else :
        replace = [0]
        classifier = [[11,-1, 1],[12,0,1]]
        divide = [14,15,17,18,20,25,28]
        squarer = [1,3,4,5,6,9,19]
        logger = [0,2,8,10,13,16,26]
        deleter = [21,22,23,29]
        nothing = [7,24,27]
    
    return replace, classifier, divide, squarer, logger, deleter,nothing
    
    
def square_root(X,col_to_sqrt):
    ''' 
    takes the square root of the elements of X in col_to_sqrt

    :param X: The numpy Matrix
    :param col_to_sqrt: The Column to take the square root from
    
    :return: The Column(s) square rooted
    :rtype: numpy matrix or array
    '''
    data_sqrted = X[:,col_to_sqrt].copy()
    data_sqrted[data_sqrted >=  0] = np.sqrt(data_sqrted[data_sqrted >= 0])
    return data_sqrted


def logarithm(X,col_to_log): 
    ''' 
    takes the log of the elements of X in col_to_log

    :param X: The numpy Matrix
    :param col_to_sqrt: The Column to take the log from
    
    :return: The Column(s) logged
    :rtype: numpy matrix or array
    '''
    
    data_loged = X[:,col_to_log].copy()
    data_loged[data_loged > 0] = np.log(data_loged[data_loged > 0])
    data_loged[data_loged == 0] = np.mean(data_loged[data_loged > 0])
    return data_loged


def classify(X,col_to_classify,class1,class2):
    ''' 
    classifies the elements of X in col_to_classify in 2 predefined classes 
    according to a threshold and returns the result.


    :param X: The numpy Matrix
    :param col_to_classify: The Column to classify
    :param class1: The first option 
    :param class2: The second option
    
    :return: The Column(s) classified 
    :rtype: numpy matrix or array
    '''
    
    data_classified = X[:,col_to_classify].copy()
    threshold =  (class1 + class2)/2.0
    data_classified[data_classified < threshold] = class1
    data_classified[data_classified >= threshold] = class2
    return data_classified


def divide_by_max(X,col_to_divide):
    ''' 
    divides the elements of X in col_to_divide by the absolute maximum 
    to bring the values between -1 and 1.

    :param X: The numpy Matrix
    :param col_to_divide: The Column to divide by the max
    
    :return: The Column(s) divides by their respective max 
    :rtype: numpy matrix or array
    '''
    
    data_divided = X[:,col_to_divide].copy()
    absolute_max = np.amax(data_divided, axis = 0)
    data_divided = data_divided / absolute_max
    return data_divided


def replace_undefined_with_mean(X,col_to_replace):
    ''' 
    Replaces the -999 values from a column by the mean of the other values of that column 

    :param X: The numpy Matrix
    :param col_to_replace: The Column to replace with the mean
    
    :return: The Column(s) with the values replaced
    :rtype: numpy matrix or array
    '''
    
    data_replaced = X[:,col_to_replace].copy()
    data_replaced[data_replaced == -999] = np.mean(data_replaced[data_replaced != -999])
    return data_replaced
    

def standardize(x):
    ''' 
    Standardizes the input using mean and std

    :param X: The numpy Matrix
    
    :return: The Matrix Standardized
    :rtype: numpy Matrix
    '''
    
    centered_data = x - np.mean(x, axis=0)
    std_data = centered_data / np.std(centered_data, axis=0)
    
    return std_data


def feature_cleaning (X,replace,divide,sqrt,log,classifier,delete,nothing):
    ''' 
    Does all the required operations on the matrix X


    :param X: The numpy Matrix we want to apply all the operations to
    :param replace: The Column(s) to replace the values in from -999 to the mean
    :param divide: The Column(s) to divide by the max values
    :param sqrt: The Column(s) to square root
    :param log: The Column(s) to log
    :param classify: The Column(s) to classify
    :param delete: The Column(s) to delete


    :return: The Matrix cleaned
    :rtype: numpy Matrix
    '''
        
    X[:,replace] =  replace_undefined_with_mean(X,replace)
    X[:,divide] =  standardize(divide_by_max(X,divide))
    X[:,sqrt] =  standardize(square_root(X,sqrt))
    X[:,log] =  standardize(logarithm(X,log))
    X[:,nothing] =  standardize(X[:,nothing])
    
    for i in range (len(classifier)):
        X[:,classifier[i][0]] =  classify(X,classifier[i][0],classifier[i][1],classifier[i][2])
    
    X = np.delete(X, delete, axis=1)
    
    return X