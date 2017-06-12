from Utilities import *


class UpSizeColorNet:

    Color_weights = None
    Color_biases = None

    def __init__(self):
        with tf.variable_scope("UpSizeColorNet"):
            # Init the model.
            self._init_model()

    def _init_model(self, model_path=None):
        
        self.Color_weights = {'W_conv1':tf.Variable(tf.truncated_normal([3,3,512,256], stddev=0.0001)),
                              'W_conv2':tf.Variable(tf.truncated_normal([3,3,256,128], stddev=0.0001)),
                              'W_conv3':tf.Variable(tf.truncated_normal([3,3,128,64], stddev=0.0001)),
                              'W_conv4':tf.Variable(tf.truncated_normal([3,3,64,64], stddev=0.0001)),
                              'W_conv5':tf.Variable(tf.truncated_normal([3,3,64,32], stddev=0.0001)),
                              'W_conv6':tf.Variable(tf.truncated_normal([3,3,32,2], stddev=0.0001))}
 
        self.Color_biases = {'b_conv1':tf.Variable(tf.truncated_normal([256], stddev=0.0001)),
                             'b_conv2':tf.Variable(tf.truncated_normal([128], stddev=0.0001)),
                             'b_conv3':tf.Variable(tf.truncated_normal([64], stddev=0.0001)),
                             'b_conv4':tf.Variable(tf.truncated_normal([64], stddev=0.0001)),
                             'b_conv5':tf.Variable(tf.truncated_normal([32], stddev=0.0001)),
                             'b_conv6':tf.Variable(tf.truncated_normal([2], stddev=0.0001))}    
       
    def build(self, input_tensor):
        
        #region Colorization Net
        Dynamic_shape = input_tensor._shape.as_list()#get Tensor shape as list of integer 
        H= Dynamic_shape[1]
        W= Dynamic_shape[2]
    
        Color_layer1 = tf.nn.relu(Conv2d(input_tensor, self.Color_weights['W_conv1'], 1) + self.Color_biases['b_conv1'])     
 
        Color_layer2 = tf.nn.relu(Conv2d(Color_layer1, self.Color_weights['W_conv2'], 1) + self.Color_biases['b_conv2'])     
    
        Color_layer2_up = tf.image.resize_nearest_neighbor(Color_layer2,[H*2,W*2])
 
        Color_layer3 = tf.nn.relu(Conv2d(Color_layer2_up, self.Color_weights['W_conv3'], 1) + self.Color_biases['b_conv3']) 
    
        Color_layer4 = tf.nn.relu(Conv2d(Color_layer3, self.Color_weights['W_conv4'], 1) + self.Color_biases['b_conv4']) 
    
        Color_layer4_up = tf.image.resize_nearest_neighbor(Color_layer4,[H*4,W*4])
 
    
        Color_layer5 = tf.nn.relu(Conv2d(Color_layer4_up, self.Color_weights['W_conv5'], 1) + self.Color_biases['b_conv5']) 
    
        Color_layer6 = tf.nn.sigmoid(Conv2d(Color_layer5, self.Color_weights['W_conv6'], 1) + self.Color_biases['b_conv6']) 
 
        Output = tf.image.resize_nearest_neighbor(Color_layer6,[H*8,W*8])

        return Output
 
 
 
