import argparse
import time

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn import linear_model
from sklearn.feature_selection import RFE
import statsmodels.api as sm

from metric.confusion_matrix import ConfusionMatrix
from metric.regressor_stats import RegressorStats
from metric.roc_curve import ROCCurve


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
parser.add_argument('-plot-roc-curve', dest='plot_roc_curve', type=str2bool, nargs='?')
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

def label_encode(df):
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

def one_hot_encode(df):
    cat_vars = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week',
                'poutcome']
    for var in cat_vars:
        cat_list = pd.get_dummies(df[var], prefix=var)
        df = df.join(cat_list)
        df = df.drop(var, axis=1)

    cols = df.columns.tolist()
    cols = cols[0:9] + cols[11:] + cols[10:11]
    df = df[cols]

    df.y = df.y.astype('category').cat.rename_categories({'no': 0, 'yes': 1}).astype(int)
    print(df)

    return df

def select_rfe_features(df):
    cols = ['euribor3m', 'job_blue-collar', 'job_housemaid', 'marital_unknown', 'education_illiterate', 'default_no',
            'default_unknown',
            'contact_cellular', 'contact_telephone', 'month_apr', 'month_aug', 'month_dec', 'month_jul', 'month_jun',
            'month_mar',
            'month_may', 'month_nov', 'month_oct', "poutcome_failure", "poutcome_success"]

    return df[cols]

def drop_logit_features(df):
    return df.drop('default_no', axis=1) \
             .drop('default_unknown', axis=1) \
             .drop('contact_cellular', axis=1) \
             .drop('contact_telephone', axis=1)

def init_dataset(args):
    print("Initializing dataset...")

    df_ds = one_hot_encode(pd.read_csv(args.dataset,header=0,sep=';'))

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

    os = SMOTE(random_state=0)
    training_set_x, training_set_y = os.fit_sample(training_set_x, training_set_y)

    # training_set_x = select_rfe_features(training_set_x)
    # training_set_x = drop_logit_features(training_set_x)
    # validation_set_x = select_rfe_features(validation_set_x)
    # validation_set_x = drop_logit_features(validation_set_x)
    # test_set_x = select_rfe_features(test_set_x)
    # test_set_x = drop_logit_features(test_set_x)
    #
    # logit_model = sm.Logit(training_set_y, training_set_x)
    # result = logit_model.fit()
    # print(result.summary2())

    # Data pre-processing
    training_set_x, training_mean, training_std = normalize(training_set_x.values)
    validation_set_x, _, _ = normalize(validation_set_x.values, training_mean, training_std)
    test_set_x, _, _ = normalize(test_set_x.values, training_mean, training_std)

    classes = ['no', 'yes']

    return classes, training_set_x, training_set_y, validation_set_x, validation_set_y, test_set_x, test_set_y

def print_stats(y_real, y_pred, data_type='Train'):
    stats = RegressorStats.get_stats(y_real, y_pred)
    print('%s accuracy: %.2f' % (data_type, stats['accuracy']))
    print('%s precision: %.2f' % (data_type, stats['precision']))
    print('%s recall: %.2f' % (data_type, stats['recall']))
    print('%s f0.5 score: %.2f' % (data_type, stats['f0.5']))
    print('%s f1 score: %.2f' % (data_type, stats['f1']))
    print('%s f2 score: %.2f' % (data_type, stats['f2']))

def main():
    args = parser.parse_args()

    start_time = time.process_time()
    classes, training_set_x, training_set_y, validation_set_x, validation_set_y, test_set_x, test_set_y = init_dataset(args)
    print("Dataset initialization time: %s seconds" % str(time.process_time() - start_time))

    model = linear_model.LogisticRegression(multi_class='multinomial', solver='sag', max_iter=3000, tol=0.000001)

    #rfe_model = RFE(model, 20)
    # Fit model
    #rfe_model = rfe_model.fit(training_set_x, training_set_y)

    model = model.fit(training_set_x, training_set_y)

    print_stats(training_set_y, model.predict(training_set_x), data_type='Train')
    print_stats(validation_set_y, model.predict(validation_set_x), data_type='Validation')
    print_stats(test_set_y, model.predict(test_set_x), data_type='Test')

    print("Execution time: %s seconds" % str(time.process_time() - start_time))

    if (args.plot_roc_curve):
        ROCCurve.plot_roc_curve(test_set_y, model.predict(test_set_x), model.predict_proba(test_set_x))
    if (args.plot_confusion_matrix):
        ConfusionMatrix.plot_confusion_matrix(test_set_y, model.predict(test_set_x), classes)


if __name__ == '__main__':
    main()