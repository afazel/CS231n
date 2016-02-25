import numpy as np
import h5py
import math

def save_data() :
    DATA_FILENAME = "/Users/azarf/Documents/Courses/Winter2016/CS231N/project/CS231n/kaggle/fer2013/fer2013.csv"
    IMG_HEIGHT = 48
    IMG_WIDTH = 48

    x_training = np.empty([28709, 48, 48])
    x_test = np.empty([3589, 48 ,48])
    x_validation = np.empty([3589, 48 ,48])
    
    y_trainig = np.empty(28709)
    y_test = np.empty(3589)
    y_validation = np.empty(3589)

    train_idx = 0
    test_idx = 0
    val_idx = 0

    with open(DATA_FILENAME, 'r') as f:
        next(f) # ignore header
        for line in f:
            kaggle_label, data_str, category = line.split(',')[:3]
            data = data_str.split(' ')
            img = np.empty(shape=(IMG_HEIGHT, IMG_WIDTH), dtype = np.uint8)
            for idx, val in enumerate(data):
                img[math.floor(idx/IMG_HEIGHT), idx % IMG_WIDTH] = float(val)

            category = category.strip()

            if category == "Training":
                print train_idx
                x_training[train_idx, :, :] = img
                y_trainig[train_idx] = int(kaggle_label)
                train_idx += 1
            elif category == "PublicTest":
                print val_idx
                x_validation[val_idx, :, :] = img
                y_validation[val_idx] = int(kaggle_label)
                val_idx += 1
            elif category == "PrivateTest":
                print test_idx
                x_test[test_idx, :, :] = img
                y_test[test_idx] = int(kaggle_label)
                test_idx += 1


    h5f_train = h5py.File('/Users/azarf/Documents/Courses/Winter2016/CS231N/project/training_data.h5', 'w')
    h5f_train.create_dataset('dataset_1', data = x_training)
    h5f_train.close()

    h5f_test = h5py.File('/Users/azarf/Documents/Courses/Winter2016/CS231N/project/test_data.h5', 'w')
    h5f_test.create_dataset('dataset_2', data = x_test)
    h5f_test.close()

    h5f_val = h5py.File('/Users/azarf/Documents/Courses/Winter2016/CS231N/project/validation_data.h5', 'w')
    h5f_val.create_dataset('dataset_3', data = x_validation)
    h5f_val.close()

    h5f_train_y = h5py.File('/Users/azarf/Documents/Courses/Winter2016/CS231N/project/training_lebels.h5', 'w')
    h5f_train_y.create_dataset('dataset_4', data = y_trainig)
    h5f_train_y.close()

    h5f_test_y = h5py.File('/Users/azarf/Documents/Courses/Winter2016/CS231N/project/test_labels.h5', 'w')
    h5f_test_y.create_dataset('dataset_5', data = y_test)
    h5f_test_y.close()

    h5f_val_y = h5py.File('/Users/azarf/Documents/Courses/Winter2016/CS231N/project/validation_labels.h5', 'w')
    h5f_val_y.create_dataset('dataset_6', data = y_validation)
    h5f_val_y.close()
     
               
save_data()
            