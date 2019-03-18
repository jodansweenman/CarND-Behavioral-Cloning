# **Behavioral Cloning** 

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center1.png "Center"
[image2]: ./examples/left1.png "Recovery Image"
[image3]: ./examples/left2.png "Recovery Image"
[image4]: ./examples/right1.png "Recovery Image"
[image5]: ./examples/right2.png "Recovery Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 64 (model.py lines 55-92) 

The model includes RELU layers to introduce nonlinearity (code lines 66, 70, and 74), and the data is normalized in the model using a Keras lambda layer (code line 59).

Each image was cropped to avoid training meaningless information and to normalize the data, runs were taken in both the clockwise and counter clockwise directions.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 82, 86, and 90). 

The model was trained and validated on different data sets on multiple runs on both directions around the track. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 96).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and repeating this process going clockwise as well around the track.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to drive in slow detailed laps in both clockwise and counterclockwise directions around the track, with 1 set being more reckless around the track, and the other being careful and slow in order to maximize the amount of variety. I tried to mess around with more complicated architecure, such as the Nvidia architecture suggested, but I had memory issues and reverted to something that is between the LeNet architecutre I was familiar with and the Nvidia architecture, but with less overall layers.

In order to seee how the model was working, I split the data into training and validation sets. Before I shuffled the data sets, my validation set was all centered around the car driving clockwise, which made my validation loss quite high. When first deriving the model, I ran into low mean squared error on the training set, but high mean squared error on the validation set due to my model overfitting overfitting. I was able to help tune this down by getting the network size and parameters more dialed in, but I believe more data would help my model fit even better.

After tuning my training model, the last step of the process was to run the simulator and drive to see how well the car was able to navigate around the first track.

In simulating my model, I was able to drive around the track at 15mph without leaving the road. Though again, I believe that if I did even more training, with more data sets, the vehicle model would improve.

#### 2. Final Model Architecture

The final model architecture (model.py lines 66-92) consisted of a convolution neural network with the following layers and layer sizes 
- 3x3 Convolution Layer with a filter depth of 32
- RELU
- Max Pooling
- 3x3 Convolution Layer with a filter depth of 48
- RELU
- Max Pooling
- 3x3 Convolution Layer with a filter depth of 64
- RELU
- Max Pooling
- Flattening
- Fully Connected Layer with an output of 128
- RELU
- Dropout with .5 probability
- Fully Connected Layer with an output of 84
- RELU
- Dropout with .5 probability
- Fully Connected Layer with an output of 32
- RELU
- Dropout with .5 probability
- Output for Steering Angle

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to correct on over steering in a given direction. These images show what a recovery looks like starting from the left side :

![alt text][image2]

This is what it looks like starting from the right side:

![alt text][image4]

Then I repeated this process several time in order to get lots of data points.


After the collection process, I had about 11300 number of data points. I then preprocessed this data by lamdas normalizaion and then cropping each image before training on them.

As stated before, I shuffled the data set and put 20% of the data into the validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the test performance on the test track. I used an adam optimizer so that manually training the learning rate wasn't necessary.


### Final Results

Be sure to look at the video in the repo, as I was able to go around the track, and my model succeeded without actually even touching the lines.
