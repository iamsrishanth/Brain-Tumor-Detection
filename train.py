import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
import cv2
import imutils
import numpy as np
import time
from os import listdir
import os

def crop_brain_contour(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        if not cnts:
            return image
        c = max(cnts, key=cv2.contourArea)
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]            
        return new_image
    except Exception:
        return image

def load_data(dir_list, image_size):
    X = []
    y = []
    image_width, image_height = image_size
    for directory in dir_list:
        if not os.path.exists(directory):
            continue
        for subclass in listdir(directory):
            subclass_dir = os.path.join(directory, subclass)
            if not os.path.isdir(subclass_dir):
                continue
            for filename in listdir(subclass_dir):
                img_path = os.path.join(subclass_dir, filename)
                image = cv2.imread(img_path)
                if image is None: continue
                image = crop_brain_contour(image)
                if image.size == 0: continue
                image = cv2.resize(image, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)
                image = image / 255.
                X.append(image)
                if subclass in ['no', 'notumor']:
                    y.append([0])
                else:
                    y.append([1])
                
    X = np.array(X)
    y = np.array(y)
    X, y = shuffle(X, y)
    print(f'Number of examples is: {len(X)}')
    print(f'X shape is: {X.shape}')
    print(f'y shape is: {y.shape}')
    return X, y

def split_data(X, y, test_size=0.2):
    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=test_size)
    X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5)
    return X_train, y_train, X_val, y_val, X_test, y_test

def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m}:{round(s,1)}"

def compute_f1_score(y_true, prob):
    y_pred = np.where(prob > 0.5, 1, 0)
    score = f1_score(y_true, y_pred)
    return score

def build_model(input_shape):
    X_input = Input(input_shape) 
    X = ZeroPadding2D((2, 2))(X_input) 
    X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X) 
    X = MaxPooling2D((4, 4), name='max_pool0')(X) 
    X = MaxPooling2D((4, 4), name='max_pool1')(X) 
    X = Flatten()(X) 
    X = Dense(1, activation='sigmoid', name='fc')(X) 
    model = Model(inputs = X_input, outputs = X, name='BrainDetectionModel')
    return model

if __name__ == '__main__':
    IMG_WIDTH, IMG_HEIGHT = (240, 240)
    
    print("Loading data...")
    X, y = load_data(['Training', 'Testing'], (IMG_WIDTH, IMG_HEIGHT))
    if len(X) == 0:
        print("No data found in 'Training' or 'Testing'.")
        exit(1)
        
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, test_size=0.3)
    
    IMG_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 3)
    model = build_model(IMG_SHAPE)
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    log_file_name = f'brain_tumor_detection_cnn_{int(time.time())}'
    tensorboard = TensorBoard(log_dir=f'logs/{log_file_name}')
    filepath="cnn-parameters-improvement-{epoch:02d}-{val_accuracy:.2f}"
    
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Save the model
    checkpoint = ModelCheckpoint("models/{}.keras".format(filepath), monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    
    print("Training model...")
    start_time = time.time()
    model.fit(x=X_train, y=y_train, batch_size=32, epochs=10, validation_data=(X_val, y_val), callbacks=[tensorboard, checkpoint])
    end_time = time.time()
    execution_time = (end_time - start_time)
    print(f"Elapsed time: {hms_string(execution_time)}")
    
    loss, acc = model.evaluate(x=X_test, y=y_test)
    print(f"Test Loss = {loss}")
    print(f"Test Accuracy = {acc}")
    
    y_test_prob = model.predict(X_test)
    f1score = compute_f1_score(y_test, y_test_prob)
    print(f"Test F1 score: {f1score}")
