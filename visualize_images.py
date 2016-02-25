import cv2
import numpy as np
import math

def get_data() :
    DATA_FILENAME = "/Users/azarf/Documents/Courses/Winter2016/CS231N/project/CS231n/kaggle/fer2013/fer2013.csv"
    EXPORT_DIRECTORY = "/Users/azarf/Documents/Courses/Winter2016/CS231N/project/CS231n/kaggle/"
    IMG_HEIGHT = 48
    IMG_WIDTH = 48

    numread = 0

    with open(DATA_FILENAME, 'r') as f, open(EXPORT_DIRECTORY+"train_labels.txt", 'w') as train_labels, open(EXPORT_DIRECTORY+"val_labels.txt", 'w') as val_labels, open(EXPORT_DIRECTORY+"test_labels.txt", 'w') as test_labels:
        next(f) # ignore header
        for line in f:
            kaggle_label, data_str, category = line.split(',')[:3]
            data = data_str.split(' ')
            img = np.empty(shape=(IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
            for idx, val in enumerate(data):
                img[math.floor(idx/IMG_HEIGHT), idx % IMG_WIDTH] = val

            category = category.strip()

            if category == "Training":
                folder_name = "train"
                output_file = EXPORT_DIRECTORY+folder_name+"/"+str(numread)+".jpg"
                train_labels.write(output_file+" "+kaggle_label+"\n")
            elif category == "PublicTest":
                folder_name = "val"
                output_file = EXPORT_DIRECTORY+folder_name+"/"+str(numread)+".jpg"
                val_labels.write(output_file+" "+kaggle_label+"\n")
            elif category == "PrivateTest":
                folder_name = "test"
                output_file = EXPORT_DIRECTORY+folder_name+"/"+str(numread)+".jpg"
                test_labels.write(output_file+" "+kaggle_label+"\n")

            output_file = EXPORT_DIRECTORY+folder_name+"/"+str(numread)+".jpg"

            print "Writing "+output_file+" emotion: "+kaggle_label

            cv2.imwrite(output_file, img)
            cv2.imshow("Image", img)
            cv2.waitKey(100)

            numread += 1

get_data()
