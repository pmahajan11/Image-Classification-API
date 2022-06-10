import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model

import warnings
warnings.filterwarnings("ignore")


model_weights_path = "model_weights/pneumonia-model-weights-acc-89.58.h5"


def preprocess_image(input_img):
    #bgr = cv2.imread(input_img)
    lab = cv2.cvtColor(input_img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[0] = clahe.apply(lab[0])
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    resized_img = cv2.resize(bgr, (375, 300)) # resize image
        
    return resized_img.reshape(-1, 300, 375, 3)



def build_model(model_weights_path):
    img_shape = (300, 375, 3)
    #model_name = 'EfficientNetB3'
    base_model = tf.keras.applications.efficientnet.EfficientNetB3(include_top=False, input_shape=img_shape, pooling='max') 
    base_model.trainable = True
    x = base_model.output
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 )(x)
    x = Dense(256, kernel_regularizer = regularizers.l2(l = 0.016),
              activity_regularizer=regularizers.l1(0.006),
              bias_regularizer=regularizers.l1(0.006), activation='relu')(x)
    x = Dropout(rate=.4, seed=123)(x)       
    output = Dense(4, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.load_weights(model_weights_path)
    
    return model


model = build_model(model_weights_path=model_weights_path)


def predict_image(image, model=model):
    pred = model.predict(preprocess_image(image))
    pred = np.round(pred[0], decimals=4)
    pred_list = pred.tolist()
    pred_dict = {"COVID-19": pred_list[0],
                 "Normal": pred_list[1],
                 "Pneumonia-Bacterial": pred_list[2],
                 "Pneumonia-Viral": pred_list[3]}
    
    return pred_dict
