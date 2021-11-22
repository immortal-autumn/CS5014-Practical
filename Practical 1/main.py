import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import sklearn.preprocessing as prep

# Global variable.
original_file = 'data/crx.data'
clean_file = 'data/clean.data'
conversed_file = 'data/conv.data'
missing_file = 'data/missing.data'
training_file = 'data/training.data'
test_file = 'data/test.data'
validation_file = 'data/valid.data'


# Clean the missing data, divide two data sets to different files.
# Pandas may not sufficient in this process, read lines directly.
def clean_missing_data():
    c = open(clean_file, 'w+')
    m = open(missing_file, 'w+')
    with open(original_file, 'r') as f:
        for line in f:
            nline = line.replace('\n', '')
            arr = nline.split(',')
            flag = True
            for i in arr:
                if i == '?':
                    m.write(line)
                    flag = False
                    break
            if flag:
                c.write(line)

    c.close()
    m.close()


def clean_duplicates():
    data = pd.read_csv(clean_file, header=None)
    data = data.drop_duplicates()
    data.to_csv(clean_file, index=False, header=None)


# Transfer data that is an object numerically, change the encoding of the data.
# Encoding uses One-Hot encoding
def data_conversion():
    data = pd.read_csv(clean_file, header=None)
    x, y = data.shape
    remove_index = []
    for i in range(0, y - 1):
        if data[i].dtype == 'object':
            remove_index.append(i)
        else:
            # Normalization
            print(data[i].head(3))
            data.loc[:, i] = prep.scale(data[i])
            # scaler = prep.MinMaxScaler()
            # data.loc[:, i] = scaler.fit_transform(data[i])
            # print(data[i].head(3))
    remove_index.reverse()

    for i in remove_index:
        df = pd.get_dummies(data[i])
        for j in df:
            data.insert(i, j, df[j], allow_duplicates=True)
    print(remove_index)
    data = data.drop(axis=1, labels=remove_index)

    data.to_csv(conversed_file, index=False, header=None)


# This function cleans the data and create the training set and the testing set.
# Since we already have 600+ rows of data, we can choose a part of the data as a training set
def sampling():
    data = pd.read_csv(conversed_file, header=None)
    train_set, test_set = train_test_split(data, test_size=0.2)
    print(f'Size for test set is: {test_set.shape}')
    # print(f'Size for train set is: {train_set.shape}')
    test_set.to_csv(test_file, index=False, header=None)
    train_set.to_csv(training_file, index=False, header=None)
    # We still need a validation set
    data = pd.read_csv(training_file, header=None)
    # 20% as validation set
    valid_set, train_set = train_test_split(data, test_size=0.8)
    train_set.to_csv(training_file, index=False, header=None)
    valid_set.to_csv(validation_file, index=False, header=None)
    print(f'Size for train set is: {train_set.shape}')
    print(f'Size for validation set is: {valid_set.shape}')
    # print(data.describe())
    # plt.show()


# Plotting and verification
def analyse():
    data = pd.read_csv(training_file, header=None)
    print(data.head(10))
    # plt.scatter(data[0], data[15])
    data[15].hist()
    plt.show()


# Function of learning, return the model
def learning():
    data = pd.read_csv(training_file, header=None)
    x, y = data.shape
    # print(data.loc[:, range(15)])
    # print(data.head(5))
    model = LogisticRegression(class_weight='balanced', penalty='l2', C=1, max_iter=5000)
    model.fit(data.loc[:, range(y - 1)], data.loc[:, y - 1])
    # print(model.classes_)
    return model


# Function of validation, avaiable for both testing and validation
def validation(data, model):
    x, y = data.shape
    Y = model.predict(data.loc[:, range(y - 1)])
    Y_res = data.loc[:, y - 1]
    print(metrics.accuracy_score(Y_res, Y))
    print(metrics.balanced_accuracy_score(Y_res, Y))
    print(metrics.confusion_matrix(Y_res, Y))
    y_score = model.decision_function(data.loc[:, range(y - 1)])
    y_proba = model.predict_proba(data.loc[:, range(y - 1)])
    disp = metrics.plot_precision_recall_curve(model, data.loc[:, range(y - 1)], Y_res, pos_label='+')
    print('Average precision score:', metrics.average_precision_score(Y_res, y_score, pos_label='+'))
    plt.show()
    # print(y_score)
    # print(metrics.precision_recall_curve([['+', '-']], y_proba))


# Test validation evaluates the performance of the model
def validation_test(model):
    data = pd.read_csv(validation_file, header=None)
    validation(data, model)


# Test the model with testing set and get hte final result..
def scoring(model):
    data = pd.read_csv(test_file, header=None)
    validation(data, model)


# The main function
if __name__ == '__main__':
    # clean_missing_data()
    # clean_duplicates()
    # data_conversion()
    # sampling()
    # analyse()
    kmodel = learning()
    # validation_test(model)
    scoring(kmodel)
