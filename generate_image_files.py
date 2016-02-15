import numpy as np
import math

def get_data():
    DATA_FILENAME = "dataset/fer2013/fer2013_original_all.csv"
    #EXPORT_DIRECTORY = "dataset/kaggle/"
    IMG_HEIGHT = 48
    IMG_WIDTH = 48


    x_train = np.empty([1,48,48])
    y_train = np.empty([1])
    x_test = np.empty([1,48,48])
    y_test =np.empty([1])
    x_val = np.empty([1,48,48])
    y_val = np.empty([1])
    kaggle_data = {}

    counter = 1
    with open(DATA_FILENAME, 'r') as f:
        next(f) # ignore header
        for line in f:
            
            kaggle_label, data_str, category = line.split(',')[:3]
            print "data #%s, emotion class:%s, category:%s" % (counter, kaggle_label, category)
            counter += 1
            data = data_str.split(' ')
            img = np.empty(shape=(1, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
            for idx, val in enumerate(data):
                img[0,math.floor(idx/IMG_HEIGHT), idx % IMG_WIDTH] = val

            if category.strip() == 'Training':
                x_train = np.append(x_train, img , axis=0)
                y_train = np.append(y_train, kaggle_label)
              
            elif category.strip() == "PublicTest":
                x_val = np.append(x_val, img , axis=0)
                y_val = np.append(y_val, kaggle_label)

            else:
                x_test = np.append(x_test, img , axis=0)
                y_test = np.append(y_test, kaggle_label)
            
    kaggle_data["X_train"] = x_train[1:,:,:]
    kaggle_data["X_test"] = x_test[1:,:,:]
    kaggle_data["X_val"] = x_val[1:,:,:]
    kaggle_data["y_train"] = y_train[1:]
    kaggle_data["y_test"] = y_test[1:]
    kaggle_data["y_val"] = y_val[1:]
    return kaggle_data


        # category = category.replace("\n","")

        # if category == "Training":
        #     folder_name = "train"
        #     output_file = EXPORT_DIRECTORY+folder_name+"/"+str(numread)+".jpg"
        #     train_labels.write(output_file+" "+kaggle_label+"\n")
        # elif category == "PublicTest":
        #     folder_name = "val"
        #     output_file = EXPORT_DIRECTORY+folder_name+"/"+str(numread)+".jpg"
        #     val_labels.write(output_file+" "+kaggle_label+"\n")
        # elif category == "PrivateTest":
        #     folder_name = "val"
        #     output_file = EXPORT_DIRECTORY+folder_name+"/"+str(numread)+".jpg"
        #     test_labels.write(output_file+" "+kaggle_label+"\n")

        # output_file = EXPORT_DIRECTORY+folder_name+"/"+str(numread)+".jpg"

        # print "Writing "+output_file+" emotion: "+kaggle_label

        # cv2.imwrite(output_file, img)
        # cv2.imshow("Image", img)
        # cv2.waitKey(100)

        # numread += 1