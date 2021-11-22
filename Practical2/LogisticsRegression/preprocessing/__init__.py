import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn import metrics
import sklearn.preprocessing as prep

'''
This class represents the training process. The class includes the following features:
1. Training models
2. Error check (Performance check)
3. Apply data to fit the model (Study)
'''


class Train:
    def __init__(self, train_set, validation_set):
        self.train_set = pd.read_csv(train_set)
        if validation_set is not None:
            self.validation_set = pd.read_csv(validation_set)
        self.color_model = None
        self.texture_model = None
        self.input_vars = []
        self.textures = []
        self.colors = []

    def one_hot_classify_in_out_result(self):
        pass

    def normal_classify_in_out_result(self):
        for i in self.train_set:
            if i not in ['color', 'texture']:
                self.input_vars.append(i)

    def random_forest_model_train(self):
        # self.model = RandomForestClassifier(max_depth=2, random_state=0)
        pass

    def logistic_model_train(self):
        print(f'Input parameters: {self.input_vars}')
        print('> Start training - Color model')
        self.color_model = LogisticRegression(class_weight='balanced', penalty='l2', C=1, max_iter=50000,
                                              solver='newton-cg', multi_class='multinomial')
        self.color_model.fit(self.train_set.loc[:, self.input_vars], self.train_set.loc[:, 'color'])
        print('> End training - Color model')

        print('> Start training - Texture model')
        self.texture_model = LogisticRegression(class_weight='balanced', penalty='l2', C=1, max_iter=50000,
                                                solver='newton-cg', multi_class='multinomial')
        self.texture_model.fit(self.train_set.loc[:, self.input_vars], self.train_set.loc[:, 'texture'])
        print('> End training - Texture model')

    def calculate_error(self):
        print('Evaluation for color model')
        col = self.color_model.predict(self.validation_set.loc[:, self.input_vars])
        col_o = self.validation_set.loc[:, 'color']
        print(len(col), len(col_o))
        tex = self.color_model.predict(self.validation_set.loc[:, self.input_vars])
        tex_o = self.validation_set.loc[:, 'texture']
        print(f'The evaluation of color model shows as follows: \n'
              f'Accuracy score: {metrics.accuracy_score(col_o, col)} \n'
              f'Balanced accuracy score: {metrics.balanced_accuracy_score(col_o, col)}\n'
              f'Confusion matrix:\n {metrics.confusion_matrix(col_o, col)}'
              f'\n\n')

        print('Evaluation for texture model')
        print(f'The evaluation of texture model shows as follows: \n'
              f'Accuracy score: {metrics.accuracy_score(tex_o, tex)} \n'
              f'Balanced accuracy score: {metrics.balanced_accuracy_score(tex_o, tex)}\n'
              f'Confusion matrix:\n {metrics.confusion_matrix(tex_o, tex)}'
              f'\n')
        return metrics.balanced_accuracy_score(col_o, col), metrics.balanced_accuracy_score(tex_o, tex)

    def predict(self, filepath):
        tmp = pd.read_csv(filepath)
        tmp = tmp.fillna(value=0)
        print(f'The shape of the prediction file is {tmp.shape}')
        # print(pd.read_csv('color_encode_config.csv'))
        color_encodes = pd.read_csv('color_encode_config.csv')['0'].values.tolist()
        texture_encodes = pd.read_csv('texture_encode_config.csv')['0'].values.tolist()
        print('Value for encoding file is', color_encodes, texture_encodes)
        pd.Series(self.color_model.predict(tmp.loc[:, self.input_vars])).\
            map(lambda x: color_encodes[x] if x < len(color_encodes) else 'Unknown').\
            to_csv("color.csv", index=False, header=False)
        pd.Series(self.texture_model.predict(tmp.loc[:, self.input_vars])).\
            map(lambda x: texture_encodes[x] if x < len(texture_encodes) else 'Unknown').\
            to_csv("texture.csv", index=False, header=False)


'''
This class represents the data preprocessing. The class includes the following:
1. Prepare data, data cleaning, data scalable and data validation
2. Separate the training set into validation set and training set
'''


class Prep:
    # Read file from the system
    def __init__(self, filename):
        self.target_set = pd.read_csv(filename)
        self.train_sets, self.valid_sets = [], []
        self.parameter = {}

    # This function divide data set into the testing set, validation set and training set
    def data_division(self, train_path, test_path):
        train_set, test_set = train_test_split(self.target_set, test_size=0.2, shuffle=True)
        print(train_set.shape, test_set.shape)
        print(self.target_set.shape)
        train_set.to_csv(train_path, index=None)
        test_set.to_csv(test_path, index=None)

    def data_fold(self):
        kf = KFold(n_splits=5, shuffle=True)
        for train_index, valid_index in kf.split(self.target_set):
            self.train_sets.append(self.target_set.loc[train_index, :])
            self.valid_sets.append(self.target_set.loc[valid_index, :])

    def extern_norm(self, filename, target_filename):
        tmp = pd.read_csv(filename)
        for i in tmp:
            for j in ['lightness_', 'redgreen_', 'blueyellow_', 'hog_', 'bimp_']:
                if j in i:
                    tmp.loc[:, i] = tmp[i].map(lambda x: (x - self.parameter[i][0]) / self.parameter[i][1])
                    break
        tmp.to_csv(target_filename, index=None)

    # Normalize:
    def data_norm(self):
        for i in self.target_set:
            for j in ['lightness_', 'redgreen_', 'blueyellow_', 'hog_', 'bimp_']:
                if j in i:
                    self.parameter[i] = [self.target_set.loc[:, i].mean(), self.target_set.loc[:, i].std()]
                    self.target_set.loc[:, i] = prep.scale(self.target_set[i])
                    break
        pd.DataFrame.from_dict(self.parameter).to_csv('data/config.csv', index=False)

    def fix_errors(self):
        print(f'Original size for the table is {self.target_set.shape}')
        # Identify value ignored - Remove useless data columns
        self.target_set = self.target_set.drop(axis=1, labels=['image', 'id', 'x', 'y'])
        # Remove duplicates and identify missing data
        self.target_set = self.target_set.drop_duplicates()
        x, y = self.target_set.shape
        indexes = []
        for i in range(x):
            for j in self.target_set:
                el = self.target_set.loc[i, j]
                if j in ['w', 'h']:
                    if el == 0:
                        indexes.append(i)
                        break
                if pd.isnull(el) or el == '?' or el == '' or el == '\n':
                    indexes.append(i)
                    break
        print(indexes)
        self.target_set = self.target_set.drop(labels=indexes)
        print(f'Updated size after clean missing is {self.target_set.shape}')

    def binary_encoding(self):
        # Encode the image
        for i in ['color', 'texture']:
            encodes = []
            for j in self.target_set.loc[:, i]:
                if j not in encodes:
                    encodes.append(j)
            self.target_set.loc[:, i] = self.target_set[i].map(lambda x: encodes.index(x))
            # print(encodes)
            pd.DataFrame(encodes).to_csv(f'{i}_encode_config.csv', index=False)

    def one_hot_encoding(self):
        # Identify unsuitable encoding - To ONE_HOT encoding - for rows [color, texture]
        df = pd.get_dummies(self.target_set['color'])
        self.target_set = pd.concat([self.target_set, df], axis=1)
        df = pd.get_dummies(self.target_set['texture'])
        self.target_set = pd.concat([self.target_set, df], axis=1)
        print(f'Updated size after concat is {self.target_set.shape}')
        self.target_set = self.target_set.drop(axis=1, labels=['color', 'texture'])
        print(f'Updated size for the table is {self.target_set.shape}')

    def save_division_table(self, valid_names, train_names):
        for i in range(5):
            self.train_sets[i].to_csv(train_names[i], index=None)
            self.valid_sets[i].to_csv(valid_names[i], index=None)

    def save_current_table(self, filename):
        self.target_set.to_csv(filename, index=None)

    def get_whole_set(self):
        return self.target_set

    def remove_indexes_in_set(self, index_list):
        self.target_set.drop(labels=index_list, axis=0)
