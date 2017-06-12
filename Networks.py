from Utilities import * 
from _1LowLevelFeatureNet import LowLevelFeatureNet
from _2MiddLevelFeatureNet import MidLevelFeatureNet
from _3GlobalLevelFeatureNet import GlobalLevelFeatureNet
from _4MultiLayer import ML
from _5ColorizationNet import UpSizeColorNet
from ReadFromDataSet import Read


class Networks:
    """construct and bulid low, mid, global, Ml, Fusion and Colorization Net Wieghts and bulid the networks"""
    
    #region Networks Data

    image_size = [224,224]
    ModelDirectory = None
    DataSet = None #hold class read from data set
    NormalizationRange = [-127, 128]

    # Low Level Network Data ----------------------------------
    Low_Level_Network = None
    LowMid = None
    LowGlobal = None
    #----------------------------------------------------------
    
    # Mid Level Network Data ----------------------------------
    Mid_level_Network = None
    Mid_output = None
    #----------------------------------------------------------
    
    # Global Level Network Data -------------------------------
    Global_level_Network = None
    global_Output = None
    #----------------------------------------------------------
    
    # ML Network Data -----------------------------------------
    ML_Network = None
    ML_Output = None
    #Output_Of_ML_NET_For_Classification = None
    #----------------------------------------------------------

    # Colorization Network Data -------------------------------
    Color_Network = None
    Color_Net_OutPut_IMAGE = None
    #----------------------------------------------------------
    
    # Fusion Network Data -------------------------------------
    Fusion_Output = None
    #----------------------------------------------------------

    #endregion

    def __init__(self, ModelDirectory):
        with tf.variable_scope("Network"):
            # Init the Networks.
            self.ModelDirectory = ModelDirectory
            self._init_model()

    def _init_model(self):

        #---------------------------------------------------------------------------------
        #intialize low, mid,glable level networks wieghts , to bulid call _build_graph(input_tensor)
        self.Low_Level_Network = LowLevelFeatureNet()
        self.Mid_level_Network = MidLevelFeatureNet()
        self.Global_level_Network = GlobalLevelFeatureNet()
        #---------------------------------------------------------------------------------
        # initialize ML Network
        self.ML_Network = ML()
        #---------------------------------------------------------------------------------
        #initialize Color Net 
        self.Color_Network = UpSizeColorNet()
        #---------------------------------------------------------------------------------
        print(" NETWORK BULIDED SUCCESSFULLY !")
        print(" ")

    def Forward(self, batch_size, Fixed_Image_As_Tensor, Dynamic_Image_As_Tensor = None):
        """ bulid all networks for feed forward process"""  

        #region Low Level Network
        # Fixed size to Global Network as an input
        # Dynamic size to Mid Network as an input
        self.LowGlobal = self.Low_Level_Network.build(Fixed_Image_As_Tensor)
        #-------------------------------------------------------------------------------------------------
        #check if that a training or testing, because in trainig we feed lowlevelnetwork with fixed size image 224 * 224
        if(Dynamic_Image_As_Tensor != None):
            self.LowMid = self.Low_Level_Network.build(Dynamic_Image_As_Tensor)
            
        else:
            self.LowMid = self.LowGlobal
        #-------------------------------------------------------------------------------------------------
        #endregion

        #====================================================================================================
    
        #region Mid Level Network
        # Input : Dynamic size of low level Network
        self.Mid_output = self.Mid_level_Network.build(self.LowMid)
        #endregion

        #====================================================================================================

        #region Global Level Network
        # Input : Fixed size of low level Network
        self.global_Output = self.Global_level_Network.build(self.LowGlobal)
        #endregion
    
        #====================================================================================================

        #region ML 
        # Input the output of Global level Network
        self.ML_Output = self.ML_Network.build(self.global_Output)
        #TODO : Edit ML to get 512 output for Classification
        #Output_Of_ML_NET_For_Classification = ML_Network.output_512
        #endregion

        #====================================================================================================

        #region Fusion Network
        # input -> output of Mid Net & output Of ML
        self.Fusion_Output = FUSE(batch_size, self.ML_Output, self.Mid_output)
        #endregion

        #====================================================================================================

        #region Colorization Network 
        #input : output of Fusion
        self.Color_Net_OutPut_IMAGE = self.Color_Network.build(self.Fusion_Output)
        #endregion
        #---------------------------------------------------------------------------------
        print("OUTPUT IMAGE shape", self.Color_Net_OutPut_IMAGE)
        #---------------------------------------------------------------------------------
        return self.Color_Net_OutPut_IMAGE

    def train(self, sess, DataSetPath, NumberOfElementInDatSet, Batch_size, EpochsNumber,LearningRate, PreTrainedFlag ):

        #initialize Data required for Training 
        self.DataSet = Read(DataSetPath)

        # PreTrainedFlag Describe if there is pre trained process to load weights from a Model Directory
        # place holder
        Input_images = tf.placeholder(dtype=tf.float32,shape=[None, self.image_size[0], self.image_size[1], 1],name="X_inputs")
        Ab_Labels_tensor = tf.placeholder(dtype=tf.float32,shape=[None, self.image_size[0], self.image_size[1], 2],name="Labels_inputs")

        #forward 
        Prediction = self.Forward(Batch_size, Input_images) 


        #calculate MSE Error
        Colorization_MSE = tf.reduce_mean((Frobenius_Norm(tf.subtract(Prediction,Ab_Labels_tensor))))

        #optimize reduce error
        #Optmizer = tf.train.AdamOptimizer().minimize(Colorization_MSE)
        Optmizer = tf.train.AdamOptimizer().minimize(Colorization_MSE)
        #Optmizer = tf.train.AdadeltaOptimizer().minimize(Colorization_MSE)

        
        #load or initialize new weights for Model Network
        saver = tf.train.Saver()

        if(PreTrainedFlag == 0): 
            sess.run(tf.global_variables_initializer()) 
        else:
            saver = tf.train.import_meta_graph(self.ModelDirectory + '.meta')
            saver.restore(sess, self.ModelDirectory)

        #Training Epochs
        for epoch in range(EpochsNumber):
            epoch_loss = 0
            CurrentBatch_indx = 1
            for i in range(int(NumberOfElementInDatSet / Batch_size)):#Over batches
                #read the next batch 
                #return ABcolorTensor
                GreyImages_Batch, AbColores_values = self.DataSet.ReadNextBatch((i * Batch_size) + 1)

                a, c = sess.run([Optmizer,Colorization_MSE],feed_dict={Input_images:GreyImages_Batch,Ab_Labels_tensor:AbColores_values})
                
                epoch_loss += c
            
            print("epoch: ",epoch + 1, ",Loss: ",epoch_loss)

        if(PreTrainedFlag == 0): 
            saver.save(sess, self.ModelDirectory)
        else:
            saver.save(sess, self.ModelDirectory,write_meta_graph=False)

    def test(self, Flag, sess, FixedGreyImages_Batch, DynamicGreyImageBatch):  #flag for Multi run to not load session multi-time and consume time

        print("dynamic", DynamicGreyImageBatch[0].shape)
        ##--------------------------------------------------------------------------------------------------------

        try:
            saver = tf.train.Saver()
            saver = tf.train.import_meta_graph(self.ModelDirectory + '.meta')
            saver.restore(sess, self.ModelDirectory)
        except IOError:
            print(" Sorry! You try to train un-pre-trained Model ! Train the Model then back for me ! i am waiting :) ...")
            return None
 
        #--------------------------------------------------------------------------------------------------------
        #prepare data for Testing Process

        Dynamic_shape = DynamicGreyImageBatch[0].shape#get Tensor shape as list of integer 

        FixedImage = tf.placeholder(dtype=tf.float32,shape=[1,224,224,1])
        DynamicImage = tf.placeholder(dtype=tf.float32,shape=[1,Dynamic_shape[0],Dynamic_shape[1],1])

        #--------------------------------------------------------------------------------------------------------
        #feed 
        Prediction = self.Forward(1, FixedImage, DynamicImage) 
        Chrominance = sess.run(Prediction,feed_dict={FixedImage:FixedGreyImages_Batch, DynamicImage:DynamicGreyImageBatch})
        #--------------------------------------------------------------------------------------------------------
        #merge the Luminance
        
        NewImg = Merge_Chrominance_Luminance(Chrominance, DynamicGreyImageBatch[0])

        plt.imshow(NewImg)
        plt.show()
        plt.imsave(r'C:\Users\omar\Desktop\CTS\test.jpg',NewImg)
        return NewImg

        






