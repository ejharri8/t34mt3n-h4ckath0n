from keras.models import model_from_json
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Activation,MaxPooling2D,Dropout,Flatten
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
#import numpy as np
from keras import optimizers
import os
from keras import backend as K
from PIL import Image
import sys

def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def inference(csv_file):
    #load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    #load weights into new model
    loaded_model.load_weights("model.h5")
    
    #... read csv file
    # df is data from csv file
    filename = csv_file
    df=pd.read_csv(filename, header = None)
    df.columns = ['0','1']
    datagen=ImageDataGenerator() 
    valid_generator = datagen.flow_from_dataframe(dataframe=df, directory="data/", x_col="0",y_col="1",
                                                class_mode="raw", target_size=(320,180), batch_size=30)
    
    loaded_model.compile(optimizers.rmsprop(lr=0.0001),loss="sparse_categorical_crossentropy", 
              metrics=["accuracy", precision, recall])
    
    print('evaluate model')
    #... use the model to do inference
    STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size
    score = loaded_model.evaluate_generator(generator = valid_generator, steps = STEP_SIZE_VALID,
                                     use_multiprocessing = False, verbose = 1 )
    
    
    # accuracy
    accuracy_output = score[1]
    
    # recall = truePositives / (truePositives + falseNegatives) backend.epislon()
    recall_output = score[2]
    
    # precision = truePositives / (truePositives + falsePositives)
    precision_output = score[3]
    
    #... compute accuracy,recall and precision score using 0.5 threshold
    return {'accuracy':accuracy_output, 'recall':recall_output, 'precision':precision_output}



if __name__ == '__main__':
    #input_file = sys.argv[1] 
    input_file = 'data/dunk_layup_test_data.csv'
    ans = inference(input_file)
    print('done\n')
    print(ans)