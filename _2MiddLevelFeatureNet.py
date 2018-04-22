from Utilities import *


class MidLevelFeatureNet:

    Mid_weights = None
    Mid_biases = None

    def __init__(self):

        with tf.variable_scope("MidLvFeatNet"):
            # Init the model.
            self._init_model()
            
    def _init_model(self):
        self.Mid_weights = {'W_conv1':tf.Variable(tf.truncated_normal([3,3,512,512], stddev=0.001)),
                            'W_conv2':tf.Variable(tf.truncated_normal([3,3,512,256], stddev=0.001))}
 
        self.Mid_biases = {'b_conv1':tf.Variable(tf.truncated_normal([512], stddev=0.001)),
                           'b_conv2':tf.Variable(tf.truncated_normal([256], stddev=0.001))}

    def build(self, input_tensor):
        
        #region Mid level Net
        #print("#     Intialize Mid level Net    #")
        
        MidLev_layer1 = tf.nn.relu(Conv2d(input_tensor, self.Mid_weights['W_conv1'], 1) + self.Mid_biases['b_conv1']) 
        MidLev_layer2 = tf.nn.relu(Conv2d(MidLev_layer1, self.Mid_weights['W_conv2'], 1) + self.Mid_biases['b_conv2']) 
 
        #endregion

        output = MidLev_layer2

        return output
