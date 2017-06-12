from Utilities import *


class LowLevelFeatureNet:
 

    Low_weights = None
    Low_biases = None

    def __init__(self):

        with tf.variable_scope("LowLvFeatNet"):
            # Init the model.
            self._init_model()

    def _init_model(self):

        self.Low_weights = {'W_conv1':tf.Variable(tf.truncated_normal([3,3,1,64], stddev=0.0001),name="Low1"),
               'W_conv2':tf.Variable(tf.truncated_normal([3,3,64,128], stddev=0.0001),name="Low2"),
               'W_conv3':tf.Variable(tf.truncated_normal([3,3,128,128], stddev=0.0001),name="Low3"),
               'W_conv4':tf.Variable(tf.truncated_normal([3,3,128,256], stddev=0.0001),name="Low4"), 
               'W_conv5':tf.Variable(tf.truncated_normal([3,3,256,256], stddev=0.0001),name="Low5"),
               'W_conv6':tf.Variable(tf.truncated_normal([3,3,256,512], stddev=0.0001),name="Low6")}
        
        self.Low_biases = {'b_conv1':tf.Variable(tf.truncated_normal([64], stddev=0.0001)),
              'b_conv2':tf.Variable(tf.truncated_normal([128], stddev=0.0001)),
              'b_conv3':tf.Variable(tf.truncated_normal([128], stddev=0.0001)),
              'b_conv4':tf.Variable(tf.truncated_normal([256], stddev=0.0001)),
              'b_conv5':tf.Variable(tf.truncated_normal([256], stddev=0.0001)),
              'b_conv6':tf.Variable(tf.truncated_normal([512], stddev=0.0001))}

    def build(self, input_tensor):
        
        #region low level Net
        #print(" #     Intialize Low level Net     #")
 
        lowLev_layer1 = tf.nn.relu(Conv2d(input_tensor, self.Low_weights['W_conv1'],2) + self.Low_biases['b_conv1']) 
        lowLev_layer2 = tf.nn.relu(Conv2d(lowLev_layer1, self.Low_weights['W_conv2'], 1) + self.Low_biases['b_conv2']) 
        lowLev_layer3 = tf.nn.relu(Conv2d(lowLev_layer2, self.Low_weights['W_conv3'], 2) + self.Low_biases['b_conv3']) 
        lowLev_layer4 = tf.nn.relu(Conv2d(lowLev_layer3, self.Low_weights['W_conv4'], 1) + self.Low_biases['b_conv4']) 
        lowLev_layer5 = tf.nn.relu(Conv2d(lowLev_layer4, self.Low_weights['W_conv5'], 2) + self.Low_biases['b_conv5']) 
        lowLev_layer6 = tf.nn.relu(Conv2d(lowLev_layer5, self.Low_weights['W_conv6'], 1) + self.Low_biases['b_conv6']) 

        output = lowLev_layer6
        #endregion


        return output

