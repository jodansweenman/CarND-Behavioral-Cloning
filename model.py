### imports
import csv
import numpy as np
import cv2
import string as str
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.layers import Dropout, Conv2D, Flatten, Dense, Lambda, Cropping2D, InputLayer
from keras.layers.pooling import MaxPooling2D
from keras.models import Model, Sequential

### get the data from the simulator desktop
data_lines = [] 

with open('/home/workspace/CarND-Behavioral-Cloning-P3/data/driving_log.csv') as data_file:
    reader = csv.reader(data_file)
    next(reader)
    for line in reader:
        data_lines.append(line)

### Append the measured angles and the images in numpy arrays and display a test image with dimensions
images = []
measure = []
#img = data_lines[0]
#source_path = img[0].replace(" ","")
#img = mpimg.imread(source_path)
#plt.imshow(img)
for line in data_lines:
    #print(line[0])
    #print(line[1])
    
    #Reading Center, left, and right images, along with the steering_center
    c_img = mpimg.imread(line[0].replace(" ",""))
    l_img = mpimg.imread(line[1].replace(" ",""))
    r_img = mpimg.imread(line[2].replace(" ",""))
    steer_ctr = float(line[3].replace(" ",""))
    
    #Correction Factor of 0.2 for the left and right steering
    corr = 0.2
    steer_left = steer_ctr + corr
    steer_right = steer_ctr - corr
    images.append(c_img)
    measure.append(steer_ctr)
    images.append(l_img)
    measure.append(steer_left)
    images.append(r_img)
    measure.append(steer_right)

X_train = np.array(images)
y_train = np.array(measure)
print(X_train.shape)
print(y_train.shape)

#Define Machine Learning Model
model = Sequential()
model.add(InputLayer(input_shape=(160, 320, 3)))

# Normalization of the image
model.add(Lambda(lambda x: (x / 255) -0.5))

# Cropping the image of the incoming layer
model.add(Cropping2D(cropping = ((70,25), (0,0))))

# Define Layers
# 3x3 Convolution Layer 32 Filter size, RELU activation
model.add(Conv2D(32, kernel_size=(3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 3x3 Convolution Layer 48 Filter size, RELU activation
model.add(Conv2D(48, kernel_size = (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 3x3 Convolution Layer 48 Filter size, RELU activation
model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening to 1-Dimensional Array
model.add(Flatten())

# Fully Connected Layer with 128 outputs, RELU activation, and Dropout of .5 probability
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))

# Fully Connected Layer with 84 outputs, RELU activation, and Dropout of .5 probability
model.add(Dense(84, activation = 'relu'))
model.add(Dropout(0.5))

# Fully Connected Layer with 32 outputs, RELU activation, and Dropout of .5 probability
model.add(Dense(32, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(1))

# Training the neural network with mean squared error loss utilizing an Adam Optimizer
# Shuffle all of the training data, using 5 epochs, splitting 20% for validation of the model.
model.compile(loss='mse', optimizer='adam')
history_object = model.fit(X_train, y_train, validation_split = 0.2,epochs = 5, shuffle = True)
model.save('model.h5')