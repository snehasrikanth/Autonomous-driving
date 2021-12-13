import numpy as np
import cv2
from tqdm import tqdm #progress time bar while predicting and measuring
import time
from scipy.special import expit  #to use activation function/ sigmoid.

maximum_steering_deg, minimum_steering_deg, sbin= 180, -180, 32
#assuming that the maximum angle needed to turn is 180 degree and -180 degree to comeback to original position.
learning_rate= 1.47 #using formula


#---------------------------------------------------------------------
#separating bins
"""
Processing of steering angles where the angles where
taken in 32 equal intervals which is considered as bin.
"""

def seperating_values(): 
  bin = np.linspace(-180,180,32) 
  return bin

  #In this function we are reading the data from the csv file 


def preprocess(csv_file):
  val=np.genfromtxt(csv_file, delimiter = ',')
  steering_angles = val[:,1]
  frame_nums = val[:,0]
  return val,steering_angles,frame_nums

#-----------------------------------------------------------------------

def gaussian_distribution(array,bin,values,temp):
    i=0
    while i<len(array):
        pos = int(np.interp(array[i],bin,values))
        temp[i][pos]=1.0
        """
        Creating Gaussian distribution with
        steering at center as 1
        60 degrees as 0.25
        30 degrees as 0.50
        0 degrees as 0.75
        Distribution being [0.25, 0.50, 0.75, 1, 0.75, 0.50, 0.25]
        assuming standard deviation to be 0.25
        """
        if  pos + 1 < sbin:
            temp[i][pos - 1],temp[i][pos + 1] = 0.75,0.75
           
            if  pos + 2 < sbin:
                temp[i][pos - 2],temp[i][pos + 2]= 0.50,0.50
                 
                if  pos + 3 < sbin:
                    temp[i][pos - 2],temp[i][pos + 2] = 0.25,0.25
        i=i+1   
#------------------------------------------------------------------------
#normalize img
'''
preprocessing the image 
1. change to BW
2.remove noise=> blur
3.resize
4.crop
''' 
thresh = 255.0
dimension= (60,60)
def image_preprocessing(image_file):
    predict_image = cv2.imread(image_file)
    # dimension=(60,60)
    # dimen = predict_image.shape
    # h = predict_image.shape[0]
    # w = predict_image.shape[1]
    # c = predict_image.shape[2]
    # print(dimen,h,w,c)
    predict_image = cv2.cvtColor(predict_image, cv2.COLOR_RGB2GRAY)
    predict_image = cv2.GaussianBlur(predict_image,(7,7),0)
    predict_image = cv2.resize(predict_image,dimension)
    predict_image = predict_image[30:,:] 
    image_vector = np.array(predict_image)/thresh
    return image_vector

#-------------------------------------------------------------------------------
#Processing of steering_angles

def train(path_to_images, csv_file):
    data,deg,frame_nums = preprocess(csv_file)
    array = np.matrix(deg).transpose()
    print("-----------------------Finding the bins-------------------------")
    bin = seperating_values()
    print(bin)
    values = np.linspace(0,31,32)
    temp= np.zeros((len(array),sbin))
    gaussian_distribution(array,bin,values,temp)
    X=[]
    print("Frame Numbers------------------")
    print(frame_nums.shape[0])
    i=0
    while i< 1500: 
        predict_image = cv2.imread(path_to_images + '/' + str(int(i)).zfill(4) + '.jpg')
        predict_image = predict_image[:, :, 2]
        predict_image = cv2.resize(predict_image, dimension) 
        predict_image = predict_image[30:,:]
        X.append(np.ravel(np.array(predict_image)/thresh))
        i=i+1
    iterations = 4000
#-------------------------------------------------------------------------

    NN = Neural_Network(Lambda=1.47)
    X = np.reshape(X,(1500,(1800)))
    for _ in tqdm(range(iterations)):
        CG = NN.computeGradients(X, temp)
        params = NN.getParams()
        gradient_descent = ((1.47*CG[:])/1500)   
        params[:] = params[:] - gradient_descent
        NN.setParams(params)
    return NN

#     Create NN:
# 1.Get gradients, weights
# 2.find gradient decent
# 3.set  

#--------------------------------------------------------------
"""
In the function we try to predict the data
present in the image. 
"""

def predict(NN, image_file):
    image_vector = image_preprocessing(image_file)
    input_X = image_vector.flatten()
    bin = np.linspace(-180,180,32) 
    yHat = NN.forward(input_X)
    arg = np.argmax(yHat)
    
    return bin[arg]
   
#---------------------------------------------------------------
class Neural_Network(object):
    def __init__(self, Lambda=0):        
        self.inputLayerSize = 1800
        self.outputLayerSize = sbin
        self.hiddenLayerSize = 128
        
        # Inititalize the weights 
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)*np.sqrt(1/(self.inputLayerSize + self.hiddenLayerSize))
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)*np.sqrt(1/(self.outputLayerSize + self.hiddenLayerSize))
        self.Lambda = Lambda


    def forward(self, X):
        # Propogate inputs though network 
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3) 
        return yHat
        
    def sigmoid(self, z):
        # Apply sigmoid activation function to scalar, vector, or matrix 
        # return 1/(1+np.exp(-z))
        return expit(z) 
    
    def sigmoidPrime(self,z):
        # Gradient of sigmoid 
        # return np.exp(-z)/((1+np.exp(-z))**2)
        return (expit(z)*(1-expit(z)))
    
    def costFunction(self, X, y):
        # Compute cost for given X,y, use weights already stored in class
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2) + (self.Lambda/2)*(np.sum(self.W1**2)+np.sum(self.W2**2))
        return J
        
    def costFunctionPrime(self, X, y):
        # Compute derivative with respect to W and W2 for a given X and y
        self.yHat = self.forward(X)
        
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)+ self.Lambda*self.W2
        
        
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2) + self.Lambda*self.W1
         
        
        return dJdW1, dJdW2
        
   
    def getParams(self):
        # Get W1 and W2 unrolled into vector
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params
        
    def setParams(self, params):
        # Set W1 and W2 using single paramater vector
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))
        
    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))
