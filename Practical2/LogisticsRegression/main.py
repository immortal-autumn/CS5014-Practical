from preprocessing import Prep, Train

# File loader initialization
path = 'data/'

training_file = f'{path}data_train.csv'
cleaned_training = f'{path}data_cleaned.csv'

testing_file = f'{path}data_test.csv'
testing_file1 = f'{path}data_test1.csv'

training_split = f'{path}training_split.csv'
testing_split = f'{path}testing_split.csv'
testing_split1 = f'{path}testing_split1.csv'

validation_files = [f'{path}valid_1.csv', f'{path}valid_2.csv', f'{path}valid_3.csv', f'{path}valid_4.csv', f'{path}valid_5.csv']
training_files = [f'{path}train_1.csv', f'{path}train_2.csv', f'{path}train_3.csv', f'{path}train_4.csv', f'{path}train_5.csv']


def data_prep():
    prep = Prep(training_file)
    prep.fix_errors()
    # prep.one_hot_encoding()
    prep.binary_encoding()
    prep.save_current_table(cleaned_training)


def data_division():
    prep = Prep(cleaned_training)
    prep.data_division(training_split, testing_split)


def data_norm_fold():
    prep = Prep(training_split)
    prep.data_norm()
    prep.data_fold()
    prep.save_division_table(validation_files, training_files)
    prep.extern_norm(testing_split, testing_split1)
    prep.extern_norm(testing_file, testing_file1)


def model_train():
    ce, te = 0, 0
    for i in range(5):
        model = Train(training_files[0], validation_files[0])
        model.normal_classify_in_out_result()
        model.logistic_model_train()
        color_err, text_err = model.calculate_error()
        ce += color_err
        te += text_err
        print(f'Model {i} have been calculated successfully!')
    print(f'Performance of the model represents as: color - {ce / 5}, texture - {te / 5}')


def res_ret():
    model = Train(cleaned_training, None)
    model.normal_classify_in_out_result()
    model.logistic_model_train()
    model.predict(testing_file1)


if __name__ == '__main__':
    # data_prep()
    # data_division()
    # data_norm_fold()
    model_train()
    # res_ret()

