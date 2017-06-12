from Utilities import *


class ML:

    _model = None
    _output_512 = None
    _output_256 = None
    MLnode_for_each_layer = [1024, 512, 256]
    MLHidden_layer = []
    layers_count = 3

    def __init__(self):
        
        with tf.variable_scope("GlobalFeatNet"):
                
            # Init the model.
            
            self._init_model()

    def _init_model(self):
        
        self.MLHidden_layer.append({'weights': tf.Variable(tf.truncated_normal([7 * 7 * 512, self.MLnode_for_each_layer[0]], stddev=0.0001)),
                                    'biases': tf.Variable(tf.truncated_normal([self.MLnode_for_each_layer[0]],stddev=0.0001))})
        
        for i in range(1, self.layers_count):
            self.MLHidden_layer.append({'weights': tf.Variable(tf.truncated_normal([self.MLnode_for_each_layer[i - 1], self.MLnode_for_each_layer[i]], stddev=0.0001)),
                                        'biases': tf.Variable(tf.truncated_normal([self.MLnode_for_each_layer[i]], stddev=0.0001))})
     
    def build(self, input_tensor):
        
        #TODO : Edit this to get 512 output for classifications

        FeatureVector = tf.reshape(input_tensor,shape=[-1,7 * 7 * 512])
        
        layers_output = tf.add(tf.matmul(FeatureVector, self.MLHidden_layer[0]['weights']), self.MLHidden_layer[0]['biases'])
        layers_output = tf.nn.relu(layers_output)
 
        for j in range(1, self.layers_count):
            layers_output = tf.add(tf.matmul(layers_output, self.MLHidden_layer[j]['weights']), self.MLHidden_layer[j]['biases'])
            layers_output = tf.nn.relu(layers_output)
 

        ML_OUTPUT = layers_output    
        
        return ML_OUTPUT
        
 
    