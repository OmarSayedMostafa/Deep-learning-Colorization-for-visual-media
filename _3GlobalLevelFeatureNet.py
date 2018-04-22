from Utilities import *


class GlobalLevelFeatureNet:
 
    Global_weights = None
    Global_biases = None

    def __init__(self):
        
        with tf.variable_scope("GlobalFeatNet"):
            # Init the model.
            self._init_model()

    def _init_model(self):
        
        self.Global_weights = {'W_conv1':tf.Variable(tf.truncated_normal([3,3,512,512], stddev=0.001)),
                               'W_conv2':tf.Variable(tf.truncated_normal([3,3,512,512], stddev=0.001)),
                               'W_conv3':tf.Variable(tf.truncated_normal([3,3,512,512], stddev=0.001)),
                               'W_conv4':tf.Variable(tf.truncated_normal([3,3,512,512], stddev=0.001))}
 
        self.Global_biases = {'b_conv1':tf.Variable(tf.truncated_normal([512], stddev=0.001)),
                              'b_conv2':tf.Variable(tf.truncated_normal([512], stddev=0.001)),
                              'b_conv3':tf.Variable(tf.truncated_normal([512], stddev=0.001)),
                              'b_conv4':tf.Variable(tf.truncated_normal([512], stddev=0.001))}

    def build(self, input_tensor):
        
        #region Global level Net

        #print("#     Intialize Global Level Net     #")
 
        GlobalLev_layer1 = tf.nn.relu(Conv2d(input_tensor, self.Global_weights['W_conv1'], 2) + self.Global_biases['b_conv1']) 
        GlobalLev_layer2 = tf.nn.relu(Conv2d(GlobalLev_layer1, self.Global_weights['W_conv2'], 1) + self.Global_biases['b_conv2']) 
        GlobalLev_layer3 = tf.nn.relu(Conv2d(GlobalLev_layer2, self.Global_weights['W_conv3'], 2) + self.Global_biases['b_conv3']) 
        GlobalLev_layer4 = tf.nn.relu(Conv2d(GlobalLev_layer3, self.Global_weights['W_conv4'], 1) + self.Global_biases['b_conv4']) 
        output = GlobalLev_layer4
        #endregion
        
        return output
