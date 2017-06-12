import tensorflow as tf
import glob
from PIL import Image
import tensorflow as tf
import glob
from PIL import Image
import numpy
import numpy as np
import math
from skimage import io, color
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#region Intialize Data and Layers weights

color_image_path = r'C:\Users\omar\Desktop\CTS'

gry_image_path = r'C:\Users\omar\Desktop\GTS'

test_path = r'C:\Users\omar\Desktop\GTS'

ground_path = r'C:\Users\omar\Desktop\CTS/'

result_Path = r'C:\Users\omar\Desktop/'
 

image_size = [224,224]
NormalizationRange = [-128, 127]

ModelDirectoryPath = r'c:\users\omar\documents\visual studio 2013\Projects\VisualMediaColorization_Yarab\VisualMediaColorization_Yarab\ModelSaver\our_model'

BatchSize = 1

ColorizationModel = None
sess = None

MainHeight = 224
MainWidth = 224
#endregion

#Matplot Function to Display Gry, Result, Ground Trouth
def Display(imageNumberInFigure, coloredImage, gryImage = None, groundTruth = None):
    #display Figure 
    plt.figure(1).set_size_inches(10, 5)

    if(imageNumberInFigure == 1):
        addSubPlot(1,[coloredImage],['Result'])

    elif(imageNumberInFigure == 2):
        addSubPlot(2, [gryImage, coloredImage],['Grey Image', 'Result'])

    elif(imageNumberInFigure == 3):
        addSubPlot(3, [gryImage, coloredImage, groundTruth],['Grey Image', 'Result', 'Ground Truth'])


    plt.tight_layout()
    plt.show()
    
def addSubPlot(NumberOfImageToShowInFigure, ImageToshow, imageTitle):

    for i in range(NumberOfImageToShowInFigure):

        plt.subplot(1, NumberOfImageToShowInFigure, i+1)
        plt.title(imageTitle[i])
        if(imageTitle[i] == 'Grey Image'):
            plt.imshow(ImageToshow[i], cmap = 'gray')
        else:
            plt.imshow(ImageToshow[i])

def saveImage(image, imagepath, imageName, extension):

    plt.imsave(imagepath+'/'+imageName+'.'+extension,image)

#----------------------------------------------------------------------------------------------
#Convolution Function
def Conv2d(inp, W ,Stride):
    """Computes a 2-D convolution over a 4-D input
    Args:
    inp: 4-D tensor
    W: Kernel
    Stride: The kenrel stride over the input
    """
    return tf.nn.conv2d(inp, W, strides=[1, Stride ,Stride, 1], padding='SAME', data_format="NHWC")
#----------------------------------------------------------------------------------------------
#Fuse 2 Tensor
def FUSE(batch_size, fixed_size_input, dynamic_size_input):
    #region try
    #---------------------------------------------------------------
    # input_tensor_1 size is [batch_size, H/8, W/8, 256].
    # input_tensor_2 size is [batch_size,256].
    # Transform the [batch_size,256] tensor to [batch_size,1,1,256] tensor.
    #---------------------------------------------------------------   
    expanded_FixedInput = tf.expand_dims(fixed_size_input, 1) #[batch_size,1,256]
    expanded_FixedInput = tf.expand_dims(expanded_FixedInput,1)     #[batch_size,1,1,256]  
    #---------------------------------------------------------------
    # Transform the [batch_size,1,1,256] tensor to [batch_size,H/8,W/8,256]  
    # tensor so that it could be concatenated to the dynamic_size_input.
    #---------------------------------------------------------------
    Dynamic_shape = dynamic_size_input._shape.as_list()#get Tensor shape as list of integer 
    Tiled_FixedSizeTensor_To_Suit_Concate_Operation = tf.tile(expanded_FixedInput, [1, Dynamic_shape[1], Dynamic_shape[2], 1])
    #---------------------------------------------------------------
    # Fuse the dynamic_size_input and fixed_size_input together, so the
    # channels size becomes 512.
    # Now the tensor is size of [batch_size, H/8, W/8, 512].
    #---------------------------------------------------------------
    FusionOutput = tf.concat(3, [dynamic_size_input, Tiled_FixedSizeTensor_To_Suit_Concate_Operation])
    #---------------------------------------------------------------
    #endregion
    #Dynamic_shape = dynamic_size_input._shape.as_list()#get Tensor shape as list of integer 
    #fixed_size_input = tf.tile(fixed_size_input,[1,Dynamic_shape[1]*Dynamic_shape[2]])
    #fixed_size_input = tf.reshape(fixed_size_input,[batch_size,Dynamic_shape[1], Dynamic_shape[2], 256])
    #FusionOutput = tf.concat(3,[dynamic_size_input,fixed_size_input])

    return FusionOutput



#----------------------------------------------------------------------------------------------
def Handle_Image_Dimension_To_BE_Devisable_BY_8(Image_To_Handle):
    
    img_shape = Image_To_Handle.size
    
    #convert image Dimension to be devisable by 8 (check networks)
    if(img_shape[0] % 8 != 0 or img_shape[1] % 8 != 0):
        Newimg_shape = [(int(img_shape[0]/8) * 8), (int(img_shape[1]/8) * 8)]
        Image_To_Handle = Image_To_Handle.resize((Newimg_shape[0],Newimg_shape[1]), Image.ANTIALIAS)

    return Image_To_Handle
#----------------------------------------------------------------------------------------------
def DeNormalize(value, MinValue, MaxValue):
        '''
        DeNormalize the input value between specific range
 
        Args:
        value = pixel value
        MinValue = Old Min value
        MaxValue = Old Max value
 
        Return:
        Normalized Value
        ''' 

        MinNormalize_val = NormalizationRange[0]
        MaxNormalize_val = NormalizationRange[1]

        value = MinNormalize_val + (((MaxNormalize_val - MinNormalize_val) * (value - MinValue)) / (MaxValue - MinValue))
        
        return value
#----------------------------------------------------------------------------------------------
def Get_Batch_Chrominance(ColorImages_Batch, Batch_Size):
    
    ''''Convert every image in the batch to LAB Colorspace and normalize each value of it between [0,1]
    Return:
    AbColores_values array [batch_size,2224,224,2] 0-> A value, 1-> B value color       
    '''       
        
    AbColores_values = np.empty((Batch_Size, MainHeight, MainWidth, 2),"float32")
    
    for indx in range(Batch_Size):  
        lab = color.rgb2lab(ColorImages_Batch[indx])    
        AbColores_values[indx,:,:,0] = Normalize(lab[:,:,1],NormalizationRange[0],NormalizationRange[1])    
        AbColores_values[indx,:,:,1] = Normalize(lab[:,:,2],NormalizationRange[0],NormalizationRange[1])

    return AbColores_values
#----------------------------------------------------------------------------------------------
def Normalize(value,MinValue,MaxValue):
    '''Normalize the input value between specific range
    Args:
    value = pixel value       
    MinValue = Normalization_Range.Min       
    MaxValue = Normalization_Range.Max       
        
    Return:
    Normalized Value       
    '''       
        
    MinNormalize_val = 0
    MaxNormalize_val = 1

    value = MinNormalize_val + (((MaxNormalize_val - MinNormalize_val) * (value - MinValue)) / (MaxValue - MinValue))

    return value
#----------------------------------------------------------------------------------------------
def Merge_Chrominance_Luminance(Chrominance, Luminance):
    
    NewImg = np.empty((Luminance.shape[0],Luminance.shape[1],3))

    for i in range(len(Luminance[:,1,0])):
      for j in range(len(Luminance[1,:,0])):
         NewImg[i,j,0]= 0 + ( (Luminance[i,j,0] - 0) * (100 - 0) / (255 - 0) ) 
          
    NewImg[:,:,1] = DeNormalize(Chrominance[0,:,:,0],0,1)
    NewImg[:,:,2] = DeNormalize(Chrominance[0,:,:,1],0,1)

    NewImg = color.lab2rgb(NewImg)

    return NewImg
#----------------------------------------------------------------------------------------------
def Frobenius_Norm(M):
    '''Calculate Frobenius Normalization using formula Sqrt( sum(each (values^2) in input) )
    Args:
    M: Input Tensor     
    '''
    return tf.reduce_sum(M ** 2) ** 0.5
#----------------------------------------------------------------------------------------------
def UserInteract_train():
    global sess, ColorizationModel
    print("")
    print("         ----  Welcome In Training Process ----           ")
    print("")
    print("             .... Loading Training Weights .... ")
    print("")
    LEARN = input(" Please Enter a learning rate =  ")
    print("")    
    examples = input(" Please Ente number of Image in Data Set Folder  =  ")    
    print("")    
    epocchh = input(" Please Ente Number Of Epochs Needed For Training Process =  ")    
    print("")    
    batch = input(" Please Specify Batch Size =  ")    
    print("")    

    ColorizationModel.train(sess, color_image_path, int(examples), int(batch), int(epocchh), LEARN, 1)
#----------------------------------------------------------------------------------------------
def UserInteract_test():
    global sess, ColorizationModel
    Desire = input(" Do you want to colorize something ?! Y/N ? ")
    
    if (Desire == 'Y' or Desire == 'y'): 
        
        GreyImagesRezied_Batch = [] 
        OriginalImage_Batch=[]    
   
        Original_gray_Img = Image.open(ground_path+'12.jpg').convert('L')    
   
        Fixed_Grey_img = Original_gray_Img.resize((224,224),Image.ANTIALIAS)       
   

        Original_gray_Img = np.asanyarray(Original_gray_Img)  
   
        Fixed_Grey_img = np.asanyarray(Fixed_Grey_img)  
   
        img_shape = Original_gray_Img.shape 
   
        Original_reshaped = Original_gray_Img.reshape(img_shape[0],img_shape[1], 1)#[H,W,1] 
   
        OriginalImage_Batch.append(Original_reshaped)#[#imgs,224,224,1] 
   
        gry_img_reshaped = Fixed_Grey_img.reshape(224, 224, 1)#[224,224,1] 
   
        GreyImagesRezied_Batch.append(gry_img_reshaped)#[#imgs,224,224,1] 
   
        
   

        print("") 
        print(" loading ...")    
        print("")    
   

        result = ColorizationModel.test(1,sess,GreyImagesRezied_Batch, OriginalImage_Batch) #0 -> flag for single test 
        #Display(3,result,gry,orig)    
        #desplay(gry,result)    
        #saveImage(result,result_Path,'result01','jpg')    
   
    else: 
        print("") 
        print(" OH !!     See you later ... bye") 
        print("") 

#----------------------------------------------------------------------------------------------
def UserInteract(Model):
    global sess, ColorizationModel
    sess = tf.InteractiveSession() 
    ColorizationModel = Model

    print("")
    print("")

    print("         ---- DEEP LEARNING COLORIZATION FOR VISUAL MEDIA ----           ")

    print("")
    print(" Enter (T) if you want to continue Training, or")
    print(" Enter (t) if you want to Test an Image, or")
    userInput = input(" Enter(X) to exit   :   ")
    if(userInput =='T'):
        UserInteract_train()
    elif(userInput =='t'):
        UserInteract_test()
    else:
        print("Exit ... ")
