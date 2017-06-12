from Utilities import *

class Read:
    """Contain all operations that apply on image on DataSet"""
    #region Image Data

    ColorchrominanceVector = []
    gryDataVector = []

    MainWidth = 224
    MainHeight = 224

    Batch_Size = 1
    color_image_path = None
    #will read image as colored and convert it as gry scale at run time
    #gry_image_path = None
    #Flag = 0 #Mean Read In Training Mode / 1 -> refere to Testing Mode
    Normalization_Range = [-128,127]
    #endregion

    def __init__(self, DataSetPath):
        self.color_image_path = DataSetPath
     
    def ReadNextBatch(self, Batch_indx): 
        '''Reads the Next (grey,Color) Batch and computes the Color_Images_batch Chrominance
        (AB colorspace values)
 
        Return:
            GreyImages_Batch: List with all Greyscale images [Batch size,224,224,1]
            ColorImages_Batch: List with all Colored images [Batch size,Colored images]
        '''

        GreyImages_Batch = []
        ColorImages_Batch = []

        for ind in range(self.Batch_Size):
           
            Colored_img = Image.open(self.color_image_path +'/'+ str(Batch_indx) + '.jpg')
            #Colored_img = Image.open(r'C:\Users\omar\Desktop\CTS\1.jpg')
            
            #--------------------------------------------------------------------------------
            if(Colored_img.width != self.MainWidth and Colored_img.height != self.MainHeight):
                Colored_img = Colored_img.resize((self.MainWidth, self.MainHeight), Image.ANTIALIAS)
            #---------------------------------------------------------------------------------
            ColorImages_Batch.append(Colored_img)
            #---------------------------------------------------------------------------------
            #need to be checked again
            #Grey_img = Image.open(gry_image_path+'/' + str(Batch_indx) + '.jpg')   
            Grey_img = Colored_img.convert('L')
            Grey_img = np.asanyarray(Grey_img) 
            img_shape = Grey_img.shape
            img_reshaped = Grey_img.reshape(img_shape[0],img_shape[1], 1)#[224,224,1]       
            GreyImages_Batch.append(img_reshaped)#[#imgs,224,224,1]
            #--------------------------------------------------------------------------


        AbColores_values = Get_Batch_Chrominance(ColorImages_Batch,self.Batch_Size) 

        return  GreyImages_Batch, AbColores_values

    def ReadFullVector(self, number_of_images):
        for i in range(1,number_of_images+1):
                
            Colored_img = Image.open(self.color_image_path +'/'+ str(i) + '.jpg')
            #--------------------------------------------------------------------------------
            if(Colored_img.width != self.MainWidth and Colored_img.height != self.MainHeight):
                Colored_img = Colored_img.resize((self.MainWidth, self.MainHeight), Image.ANTIALIAS)
            #---------------------------------------------------------------------------------
            self.ColorchrominanceVector.append(Colored_img)
            #---------------------------------------------------------------------------------
            #need to be checked again   
            Grey_img = Colored_img.convert('L')
            Grey_img = np.asanyarray(Grey_img) 
            img_shape = Grey_img.shape
            img_reshaped = Grey_img.reshape(img_shape[0],img_shape[1], 1)#[224,224,1]       
            self.gryDataVector.append(img_reshaped)#[#imgs,224,224,1]
      



