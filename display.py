import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image as i
def imageClass(i):
    i = i+65
    return chr(i)

IMG_DIM1 = 30
IMG_DIM2 = 30

from two_pass2 import hk
def segment(image_file):
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    cv2.imshow("win", image)
    cv2.waitKey()
    # image = cv2.GaussianBlur(image, (3, 3), 0)
    thresh, image = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY)
    # thresh, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    objs = hk(image)
    return (image, objs)

def preprocess_cnn(img):
    img = cv2.resize(img, (IMG_DIM1,IMG_DIM2))
    img = cv2.bitwise_not(img)
    image = i.img_to_array(img)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = np.expand_dims(image, axis=0)
    #img = img/255.0
    # img = img.reshape(1, IMG_DIM1, IMG_DIM2, 3)
    return image


model_file = "model14.json"
weights_file = "BestWeights14.h5"
json_file = open(model_file, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(weights_file)
print("Loaded model from disk")

def predict_class(img):
    global loaded_model
    img = preprocess_cnn(img)
    result = loaded_model.predict(img)
    return imageClass(np.argmax(result[0]))

#image_file = "res1.png"
def process_detect(image_file):
    img, objs = segment(image_file)
    pred_list = []
    for obj in objs:
        # cv2.imshow("win", img[obj[0]:obj[1], obj[2]:obj[3]])
        # cv2.waitKey()
        pred_list.append(predict_class(img[obj[0]:obj[1], obj[2]:obj[3]]))

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    i = 0
    for obj in objs:
        cv2.rectangle(img, (obj[2],obj[0]), (obj[3],obj[1]), color=(0,255,0), thickness=2)
        cv2.putText(img, pred_list[i], (obj[2]+5, obj[0]), 2, 0.7, (0, 0, 255))
        i+=1
    cv2.imshow("win", img)
    cv2.waitKey()

image_file = input("Enter file name:")
process_detect(image_file)