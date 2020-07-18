import argparse
import time

import numpy as np
import pandas as pd
from pandas import Index
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from methods.cost_calculus import CostCalculus
from methods.logistic_regressor import LogisticRegressor
from metric.confusion_matrix import ConfusionMatrix
from metric.predict_regressor import PredictRegressor
from metric.regressor_stats import RegressorStats
from methods.scikit_regressor import ScikitRegressor


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='Logistic Regression.')
parser.add_argument('-dataset', dest='dataset')
parser.add_argument('-plot-confusion-matrix', dest='plot_confusion_matrix', type=str2bool, nargs='?')
parser.add_argument('-plot-error', dest='plot_error', type=str2bool, nargs='?')

FRAC_VALIDATION = 0.2
FRAC_TEST = 0.1

def normalize(df_values, mean=None, std=None):

    # Compute mean and standard deviation
    if mean is None:
        mean = np.mean(df_values, axis=0)
    if std is None:
        sum = np.sum(df_values, axis=0)
        std = np.sqrt(np.sum((sum - mean) ** 2) / (df_values.shape[0]*df_values.shape[1] - 1))
        #std = np.std(df_values.astype(float), axis=0)

    # Normalization
    for i in range(len(df_values)):
        df_values[i] = (df_values[i] - mean)/std

    return df_values, mean, std

def print_stats(y_real, y_pred, data_type='Train'):
    stats = RegressorStats.get_stats(y_real, y_pred)
    print('%s accuracy: %.2f' % (data_type, stats['accuracy']))
    print('%s precision: %.2f' % (data_type, stats['precision']))
    print('%s recall: %.2f' % (data_type, stats['recall']))
    print('%s f0.5 score: %.2f' % (data_type, stats['f0.5']))
    print('%s f1 score: %.2f' % (data_type, stats['f1']))
    print('%s f2 score: %.2f' % (data_type, stats['f2']))

def print_regressors(regressors):
    for regressor in regressors:
        print(regressor['regressor'])

def logistic_regression_one_vs_all(args, classes, train_set_x, train_set_y, val_set_x, val_set_y, test_set_x, test_set_y):
    print("Starting Logistic Regression One-vs-All...")
    val = input('Set maximum iterations (default: 100): ')
    max_iterations = 100
    if val != '':
        max_iterations = int(val)
    val = input('Set learning rate (default: 0.01): ')
    learning_rate = 0.01
    if val != '':
        learning_rate = float(val)
    val = input('Set tolerance (default: 0.000001): ')
    tolerance = 0.000001
    if val != '':
        tolerance = float(val)

    start_time = time.process_time()

    regressors = LogisticRegressor.regressor(train_set_x, train_set_y, val_set_x, val_set_y, max_iterations,
                                             learning_rate, tolerance, method='bgd', type='onevsall')

    print('\nLogistic Regressor One-vs-All:')
    print('Coefficients (model): ')
    print_regressors(regressors)
    print('Final mean train error (all classes): \n', RegressorStats.mean_error(regressors, 'train_error'))
    print('Final mean validation error (all classes): \n', RegressorStats.mean_error(regressors, 'val_error'))
    print_stats(train_set_y, PredictRegressor.predict(regressors, train_set_x), data_type='Train')
    print_stats(val_set_y, PredictRegressor.predict(regressors, val_set_x), data_type='Validation')
    print_stats(test_set_y, PredictRegressor.predict(regressors, test_set_x), data_type='Test')

    print("Execution time: %s seconds" % str(time.process_time() - start_time))

    if (args.plot_confusion_matrix):
        ConfusionMatrix.plot_confusion_matrix(test_set_y, PredictRegressor.predict(regressors, test_set_x),
                                              classes)

    if(args.plot_error):
        CostCalculus.plot_error_logistic(regressors, classes)

def multinomial_logistic_regression(args, classes, train_set_x, train_set_y, val_set_x, val_set_y, test_set_x, test_set_y):
    print("Starting Multinomial Logistic Regression...")
    val = input('Set maximum iterations (default: 100): ')
    max_iterations = 100
    if val != '':
        max_iterations = int(val)
    val = input('Set learning rate (default: 0.01): ')
    learning_rate = 0.01
    if val != '':
        learning_rate = float(val)
    val = input('Set tolerance (default: 0.000001): ')
    tolerance = 0.000001
    if val != '':
        tolerance = float(val)

    start_time = time.process_time()

    regressors = LogisticRegressor.regressor(train_set_x, train_set_y, val_set_x, val_set_y, max_iterations, learning_rate, tolerance,
              method='bgd', type='multinomial')

    print('\nMultinomial Logistic Regressor:')
    print('Coefficients (model): \n', regressors[0]['regressor'])
    print('Final Train error: \n', regressors[0]['train_error'][regressors[0]['final_iteration']])
    print('Final validation error: \n', regressors[0]['val_error'][regressors[0]['final_iteration']])
    print_stats(train_set_y, PredictRegressor.predict(regressors, train_set_x, type='multinomial'), data_type='Train')
    print_stats(val_set_y, PredictRegressor.predict(regressors, val_set_x, type='multinomial'), data_type='Validation')
    print_stats(test_set_y, PredictRegressor.predict(regressors, test_set_x, type='multinomial'), data_type='Test')

    print("Execution time: %s seconds" % str(time.process_time() - start_time))

    if (args.plot_confusion_matrix):
        ConfusionMatrix.plot_confusion_matrix(test_set_y,
                                              PredictRegressor.predict(regressors, test_set_x, type='multinomial'),
                                              classes)

    if(args.plot_error):
        CostCalculus.plot_error_softmax(regressors)

def scikit_ovr_logistic_regression(args, classes, train_set_x, train_set_y, val_set_x, val_set_y, test_set_x, test_set_y):
    print("Starting Scikit Logistic Regression...")
    val = input('Set maximum iterations (default: 100): ')
    max_iterations = 100
    if val != '':
        max_iterations = int(val)
    val = input('Set tolerance (default: 0.000001): ')
    tolerance = 0.000001
    if val != '':
        tolerance = float(val)

    start_time = time.process_time()

    model = ScikitRegressor.ovr_regressor(train_set_x, train_set_y.values, max_iterations, tolerance)

    print('\nLogistic Regressor One-vs-All Scikit Learn:')
    print('Coefficients (model): \n', model.coef_)
    print('Intercept: \n', model.intercept_)
    print_stats(train_set_y, model.predict(train_set_x), data_type='Train')
    print_stats(val_set_y, model.predict(val_set_x), data_type='Validation')
    print_stats(test_set_y, model.predict(test_set_x), data_type='Test')

    print("Execution time: %s seconds" % str(time.process_time() - start_time))

    if (args.plot_confusion_matrix):
        ConfusionMatrix.plot_confusion_matrix(test_set_y, model.predict(test_set_x), classes)


def scikit_multinomial_logistic_regression(args, classes, train_set_x, train_set_y, val_set_x, val_set_y, test_set_x, test_set_y):
    print("Starting Scikit Logistic Regression...")
    val = input('Set maximum iterations (default: 100): ')
    max_iterations = 100
    if val != '':
        max_iterations = int(val)
    val = input('Set tolerance (default: 0.000001): ')
    tolerance = 0.000001
    if val != '':
        tolerance = float(val)

    start_time = time.process_time()

    model = ScikitRegressor.multinomial_regressor(train_set_x, train_set_y.values, max_iterations, tolerance)

    print('\nMultinomial Logistic Regressor Scikit Learn:')
    print('Coefficients (model): \n', model.coef_)
    print('Intercept: \n', model.intercept_)
    print_stats(train_set_y, model.predict(train_set_x), data_type='Train')
    print_stats(val_set_y, model.predict(val_set_x), data_type='Validation')
    print_stats(test_set_y, model.predict(test_set_x), data_type='Test')

    print("Execution time: %s seconds" % str(time.process_time() - start_time))

    if (args.plot_confusion_matrix):
        ConfusionMatrix.plot_confusion_matrix(test_set_y, model.predict(test_set_x), classes)

def number_labels(df):
    # le = LabelEncoder()
    # ohe = OneHotEncoder()
    #
    # le.fit(df)
    # ohe.fit(df)
    #
    # print(le.transform(df))
    # print(ohe.transform(df))

    df.job = df.job.astype('category').cat.rename_categories({
        'admin.': 1, 'blue-collar': 2, 'entrepreneur': 3, 'housemaid': 4, 'management': 5,
       'retired': 6, 'self-employed': 7, 'services': 8, 'student': 9, 'technician': 10,
       'unemployed': 11, 'unknown': 12
    }).astype(int)
    df.marital = df.marital.astype('category').cat.rename_categories({
        'divorced': 1, 'married': 2, 'single': 3, 'unknown': 4
    }).astype(int)
    df.education = df.education.astype('category').cat.rename_categories({
        'basic.4y': 1, 'basic.6y': 2, 'basic.9y': 3, 'high.school': 4, 'illiterate': 5,
       'professional.course': 6, 'university.degree': 7, 'unknown': 8
    }).astype(int)
    df.default = df.default.astype('category').cat.rename_categories({
        'no': 0, 'unknown': 2, 'yes': 1
    })
    df.housing = df.housing.astype('category').cat.rename_categories({
        'no': 0, 'unknown': 2, 'yes': 1
    }).astype(int)
    df.loan = df.loan.astype('category').cat.rename_categories({
        'no': 0, 'unknown': 2, 'yes': 1
    }).astype(int)
    df.contact = df.contact.astype('category').cat.rename_categories({
        'cellular': 1, 'telephone': 2
    }).astype(int)
    df.month = df.month.astype('category').cat.rename_categories({
        'apr': 4, 'aug': 8, 'dec': 12, 'jul': 7, 'jun': 6, 'mar': 3, 'may': 5, 'nov': 11, 'oct': 10, 'sep': 9
    }).astype(int)
    df.day_of_week = df.day_of_week.astype('category').cat.rename_categories({
        'fri': 5, 'mon': 1, 'thu': 2, 'tue': 4, 'wed': 3
    }).astype(int)
    df.poutcome = df.poutcome.astype('category').cat.rename_categories({
        'failure': 0, 'nonexistent': 2, 'success': 1
    }).astype(int)
    df.y = df.y.astype('category').cat.rename_categories({'no': 0, 'yes': 1}).astype(int)
    print(df)

    return df

def init_dataset(args):
    print("Initializing dataset...")

    df_ds = number_labels(pd.read_csv(args.dataset,header=0,sep=';'))
    test_set = df_ds.sample(frac=FRAC_TEST, random_state=1)
    df_train = df_ds.drop(test_set.index)

    print('Training DF dimensions (', (1 - FRAC_TEST) * 100.0, '% ):', df_train.shape)
    print('Test set dimensions (', FRAC_TEST * 100.0, '% ):', test_set.shape)

    # Split training data in training and validation
    validation_set = df_train.sample(frac=FRAC_VALIDATION, random_state=1)
    training_set = df_train.drop(validation_set.index)

    print('Training set dimensions (', (1 - FRAC_VALIDATION) * 100.0, '% ):', training_set.shape)
    print('Validation set dimensions (', FRAC_VALIDATION * 100.0, '% ):', validation_set.shape)

    # Split training set in variables(x) and target(y)
    training_set_x = training_set.iloc[:, :-1]
    training_set_y = training_set.iloc[:, -1]

    # Split validation set in variables(x) and target(y)
    validation_set_x = validation_set.iloc[:, :-1]
    validation_set_y = validation_set.iloc[:, -1]

    # Split validation set in variables(x) and target(y)
    test_set_x = test_set.iloc[:, :-1]
    test_set_y = test_set.iloc[:, -1]

    # Data pre-processing
    training_set_x, training_mean, training_std = normalize(training_set_x.values)
    validation_set_x, _, _ = normalize(validation_set_x.values, training_mean, training_std)
    test_set_x, _, _ = normalize(test_set_x.values, training_mean, training_std)

    classes = ['no', 'yes']

    return classes, training_set_x, training_set_y, validation_set_x, validation_set_y, test_set_x, test_set_y


def main():
    args = parser.parse_args()

    start_time = time.process_time()
    classes, training_set_x, training_set_y, validation_set_x, validation_set_y, test_set_x, test_set_y = init_dataset(args)
    print("Dataset initialization time: %s seconds" % str(time.process_time() - start_time))

    print('Choose your method:')
    print('1 - Logistic Regression One-vs-All')
    print('2 - Multinomial Logistic Regression')
    print('3 - Scikit Logistic Regression One-vs-All')
    print('4 - Scikit Multinomial Logistic Regression')
    print('Anyone - Exit')

    option = int(input('Option: '))

    if option == 1:
        logistic_regression_one_vs_all(args, classes, training_set_x, training_set_y, validation_set_x, validation_set_y,
                                       test_set_x, test_set_y)
    elif option == 2:
        multinomial_logistic_regression(args, classes, training_set_x, training_set_y, validation_set_x, validation_set_y,
                                        test_set_x, test_set_y)
    elif option == 3:
        scikit_ovr_logistic_regression(args, classes, training_set_x, training_set_y, validation_set_x, validation_set_y,
                                       test_set_x, test_set_y)
    elif option == 4:
        scikit_multinomial_logistic_regression(args, classes, training_set_x, training_set_y, validation_set_x, validation_set_y,
                                               test_set_x, test_set_y)


if __name__ == '__main__':
    main()
