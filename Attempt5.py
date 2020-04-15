import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import cmath
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Load Data
from numpy import genfromtxt

allData = genfromtxt(r"\Users\user1\Desktop\DiplomaThesis-Tzikas\training;.csv",
                     delimiter=';')  # First three columns are desired and undesired signal directions. The other columns are the antenna weights
numOfSignals = 3  # Consider only three DoAs
numOfElements = 11 # Consider antenna of 11 elements
# Randomly shuffle the data and create training and testing data
np.random.shuffle(allData)
trainDataX = allData[0:(int)(0.999 * allData.shape[0]), 0:numOfSignals]
trainDataY = allData[0:(int)(0.999 * allData.shape[0]), numOfSignals:]
testDataX = allData[(int)(0.999 * allData.shape[0]):(int)(0.9995 * allData.shape[0]), 0:numOfSignals]
testDataY = allData[(int)(0.999 * allData.shape[0]):(int)(0.9995 * allData.shape[0]), numOfSignals:]
compDataX = allData[(int)(0.9995 * allData.shape[0]):(int)(1.0 * allData.shape[0]), 0:numOfSignals]
compDataY = allData[(int)(0.9995 * allData.shape[0]):(int)(1.0 * allData.shape[0]), numOfSignals:]



def toInputImage(directions, nrows, ncols):
    maxAngle = np.max(allData[:, 0:numOfSignals])
    minAngle = np.min(allData[:, 0:numOfSignals])
    by = (maxAngle - minAngle) / (nrows * ncols)
    indices = np.zeros((numOfSignals, 2))
    image = np.zeros([nrows, ncols])

    t=0
    for i in range(len(directions)):
        index = (int)(np.floor((directions[i] - minAngle) / by))
        if index < nrows*ncols:
            image[np.unravel_index(index, (nrows, ncols), order="C")] = 1.0 if i == 0 else -1.0
            indices[t, :] = np.unravel_index(index, (nrows, ncols), order="C")
        else:
            image[np.unravel_index(index-1, (nrows, ncols), order="C")] = 1.0 if i == 0 else -1.0
            indices[t, :] = np.unravel_index(index-1, (nrows, ncols), order="C")
        t = t+1

    for m in range(nrows):
        for n in range(ncols):
            if (image[m,n]!=1.0 and image[m,n]!=-1.0):
                for k in range(numOfSignals):
                    distance = np.sqrt(np.absolute(indices[k, 0]-m)*np.absolute(indices[k, 0]-m)+np.absolute(indices[k, 1]-n)*np.absolute(indices[k, 1]-n))
                    if (k==0):
                        image[m,n] += 1/distance
                    else:
                        image[m,n] -= 1/distance

    result=np.zeros([1, 1, nrows, ncols])
    result[0,0,:,:]=image
    return result

def multComplexVectorTensors(real1, imag1, real2, imag2):
    res=torch.zeros(real2.shape[1])
    for j in range(real2.shape[1]):
    # iterate through rows of Y
        realPart=torch.zeros(1)
        imagPart=torch.zeros(1)
        for k in range(real2.shape[0]):
            realPart += real1[k]*real2[k][j] - imag1[k]*imag2[k][j]
            imagPart += imag1[k]*real2[k][j] - real1[k]*imag2[k][j]
        res[j] = torch.sqrt(torch.pow(realPart,2)+torch.pow(imagPart,2))
    return res

def multComplexVectors(real1, imag1, real2, imag2):
    res=np.zeros(real2.shape[1])
    for j in range(real2.shape[1]):
    # iterate through rows of Y
        realPart=np.zeros(1)
        imagPart=np.zeros(1)
        for k in range(real2.shape[0]):
            realPart += real1[k]*real2[k][j] - imag1[k]*imag2[k][j]
            imagPart += imag1[k]*real2[k][j] - real1[k]*imag2[k][j]
        res[j] = np.sqrt(np.power(realPart,2)+np.power(imagPart,2))
    return res

def radiationPattern( w_MV_R, w_MV_I ):
    thetasDiagram = np.arange(0, math.pi+math.pi/1800, math.pi/1800)
    a_thetasR = np.zeros((numOfElements, len(thetasDiagram)))
    a_thetasI = np.zeros((numOfElements, len(thetasDiagram)))
    for m in range(numOfElements):
        for i in range(thetasDiagram.shape[0]):
            a_thetasR[m,i] = (math.e ** (1j*math.pi*(m)*math.cos(thetasDiagram[i]))).real
            a_thetasI[m,i] = (math.e ** (1j*math.pi*(m)*math.cos(thetasDiagram[i]))).imag

    AF = multComplexVectorTensors(w_MV_R, w_MV_I, torch.Tensor(a_thetasR), torch.Tensor(a_thetasI))
    return AF

def lossFunction(trueWeights_R, trueWeights_I, predWeights_R, predWeights_I, dir):
    # Create two objective patterns and compare them for the loss function
    truePatternT = radiationPattern(trueWeights_R, trueWeights_I)
    truePatternT = truePatternT/torch.max(truePatternT)
    predPatternT = radiationPattern(predWeights_R, predWeights_I)
    predPatternT = predPatternT/torch.max(predPatternT)
    
    # Here is the difference with regard to Attempt1
    truePatternN = truePatternT.detach().numpy()
    predPatternN = predPatternT.detach().numpy()

    indicesPeaksTrue = find_peaks(truePatternN)[0]
    indicesPeaksPred = find_peaks(predPatternN)[0]
    indexMajorLobeTrue = find_nearest(indicesPeaksTrue, (dir[0]*math.pi/180)/(math.pi/1800))

    indicesNullsTrue = find_peaks(-truePatternN)[0]
    indicesNullsPred = find_peaks(-predPatternN)[0]

    indicesWantedNullsTrue = np.zeros((1,numOfSignals-1))
    for m in range(numOfSignals-1):
        indicesWantedNullsTrue[0,m] = find_nearest(indicesNullsTrue, (dir[m+1]*math.pi/180)/(math.pi/1800))


    loss = torch.zeros(1)
    loss = torch.pow(truePatternT[(int)(indexMajorLobeTrue)]-predPatternT[(int)(indexMajorLobeTrue)], 2)
    for m in range(numOfSignals-1):
        loss = loss + 4*torch.pow(truePatternT[(int)(indicesWantedNullsTrue[0,m])]-predPatternT[(int)(indicesWantedNullsTrue[0,m])], 2)
    return loss


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

 
#create the Neural Network based on Fast YOLO (Tiny-YOLO v2, )
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 16 output channels, 3x3 square volume convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv7 = nn.Conv2d(512, 1024, 3, padding=1)
        self.conv8 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.conv9 = nn.Conv2d(1024, 1024, 3, padding=1)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(1024 * 7 * 7, 256) 
        self.fc2 = nn.Linear(256, 4096)
        self.fc3 = nn.Linear(4096, 22)

    def forward(self, x):
        x = self.conv1(x) #Pass through first conv layer
        x = F.leaky_relu(x, negative_slope=0.01, inplace=False) #Activation of first layer
        
        x = F.max_pool2d(x, 2, stride=2, ceil_mode=True)
        
        x = self.conv2(x)
        x = F.leaky_relu(x, negative_slope=0.01, inplace=False)
        
        x = F.max_pool2d(x, 2, stride=2, ceil_mode=True)
        
        x = self.conv3(x)
        x = F.leaky_relu(x, negative_slope=0.01, inplace=False)
        
        x = F.max_pool2d(x, 2, stride=2, ceil_mode=True)
        
        x = self.conv4(x)
        x = F.leaky_relu(x, negative_slope=0.01, inplace=False)
        
        x = F.max_pool2d(x, 2, stride=2, ceil_mode=True)
        
        x = self.conv5(x)
        x = F.leaky_relu(x, negative_slope=0.01, inplace=False)
        
        x = F.max_pool2d(x, 2, stride=2, ceil_mode=True)
        
        x = self.conv6(x)
        x = F.leaky_relu(x, negative_slope=0.01, inplace=False)
        
        x = F.max_pool2d(x, 2, stride=2, ceil_mode=True)
        
        x = self.conv7(x)
        x = F.leaky_relu(x, negative_slope=0.01, inplace=False)
        
        x = self.conv8(x)
        x = F.leaky_relu(x, negative_slope=0.01, inplace=False)
         
        x = self.conv9(x)
        
        x = torch.tanh(self.fc1(x.view(-1, 1024*7*7)))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()

net = net.float() # Due to error stating that weights are not double

# The optimizer may change due to smoothness and other characteristics of loss function
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.0) # If training error rises and falls reduce learning rate

batch_size = 50
trainLoss = np.zeros((int)(trainDataX.shape[0]/batch_size)) # Train loss for every batch 
testLoss = np.zeros((int)(trainDataX.shape[0]/(50*batch_size))) # Loss in test set for weights updated after every batch

trainLossPerBatch = np.zeros((int)(trainDataX.shape[0]/batch_size))
pointer=0
printTestError_Index=1
a=0
for epoch in range(1):  # loop over the dataset multiple times

    running_loss = 0.0
    for j in range((int)(trainDataX.shape[0]/batch_size)): # Loop over batches 
        running_loss = 0.0
        batch_loss = torch.zeros(1)
        for i in range(batch_size): # Loop over each data point (use SGD for gradient)
            inputs=torch.Tensor(toInputImage(trainDataX[j*batch_size+i, :], nrows=448, ncols=448))
            #inputs=inputs.float()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs.float())

            weightsR = torch.zeros(numOfElements)
            weightsI = torch.zeros(numOfElements)
            index=0
            for l in range(numOfElements):
                weightsR[l] = outputs[0, l+index]
                weightsI[l] = outputs[0, l+1+index]
                index +=1

            trueWeightsR = np.zeros(numOfElements)
            trueWeightsI = np.zeros(numOfElements)
            index=0
            for t in range(numOfElements):
                trueWeightsR[t] = trainDataY[j*batch_size+i, t+index]
                trueWeightsI[t] = trainDataY[j*batch_size+i, t+1+index]
                index +=1
            error = nn.MSELoss()
            loss = error(weightsR, torch.Tensor(trueWeightsR))+error(weightsI, torch.Tensor(trueWeightsI))
            batch_loss += loss/batch_size
        batch_loss.backward()
        optimizer.step()

        # print statistics
        running_loss += batch_loss.item()
        trainLossPerBatch[pointer] = batch_loss.item()
        pointer = pointer + 1
        print(j, running_loss)
        trainLoss[j]=running_loss

        # Find error in test set after training in specific batch
        with torch.no_grad():
            if printTestError_Index%100==1:
                test_loss=0
                for k in range(testDataX.shape[0]):
                    inputs=torch.Tensor(toInputImage(testDataX[k, :], nrows=448, ncols=448))
                    #inputs=inputs.double()

                    # forward 
                    outputs = net(inputs) 
                    weightsR = torch.zeros(numOfElements)
                    weightsI = torch.zeros(numOfElements)
                    index=0
                    for l in range(numOfElements):
                        weightsR[l] = outputs[0, l+index]
                        weightsI[l] = outputs[0, l+1+index]
                        index += 1

                    trueWeightsR = np.zeros(numOfElements)
                    trueWeightsI = np.zeros(numOfElements)
                    index=0
                    for t in range(numOfElements):
                        trueWeightsR[t] = testDataY[k, t+index]
                        trueWeightsI[t] = testDataY[k, t+1+index]
                        index +=1

                    loss = lossFunction(torch.Tensor(trueWeightsR), torch.Tensor(trueWeightsI), weightsR, weightsI, testDataX[k, :]).numpy()
                    print(k, loss)
                    test_loss=test_loss+loss
                testLoss[a]=test_loss/testDataX.shape[0]
                print(a, " Test Loss ", test_loss/testDataX.shape[0])
                a = a+1
            printTestError_Index=printTestError_Index+1
            print(j, " Train Loss ", running_loss)
                
        

print('Finished Training')

#Store model
torch.save(net, r"\Users\user1\Desktop\DiplomaThesis-Tzikas\TinyYoloDoA5.pth") 

# Check angle differences [deg]
diffDesired = np.zeros(compDataX.shape[0])
diffUndesired1 = np.zeros(compDataX.shape[0])
diffUndesired2 = np.zeros(compDataX.shape[0])
for k in range(compDataX.shape[0]):
    inputs=torch.Tensor(toInputImage(compDataX[k, :], nrows=448, ncols=448))
    #inputs=inputs.double()

    # forward 
    outputs = net(inputs) 
    weightsR = torch.zeros(numOfElements)
    weightsI = torch.zeros(numOfElements)
    index=0
    for l in range(numOfElements):
        weightsR[l] = outputs[0, l+index]
        weightsI[l] = outputs[0, l+1+index]
        index += 1

    trueWeightsR = np.zeros(numOfElements)
    trueWeightsI = np.zeros(numOfElements)
    index=0
    for t in range(numOfElements):
        trueWeightsR[t] = compDataY[k, t+index]
        trueWeightsI[t] = compDataY[k, t+1+index]
        index +=1

    thetasDiagram = np.arange(0, math.pi+math.pi/1800, math.pi/1800)
    a_thetasR = np.zeros((numOfElements, len(thetasDiagram)))
    a_thetasI = np.zeros((numOfElements, len(thetasDiagram)))
    for m in range(numOfElements):
        for i in range(thetasDiagram.shape[0]):
            a_thetasR[m,i] = (math.e ** (1j*math.pi*(m)*math.cos(thetasDiagram[i]))).real
            a_thetasI[m,i] = (math.e ** (1j*math.pi*(m)*math.cos(thetasDiagram[i]))).imag
    AFtrue = multComplexVectors(trueWeightsR, trueWeightsI, a_thetasR, a_thetasI)
    AFpred = multComplexVectors(weightsR.detach().numpy(), weightsI.detach().numpy(), a_thetasR, a_thetasI)

    #Find major lobe
    indexMajorLobeTrue = find_nearest(find_peaks(AFtrue)[0], compDataX[k, 0]*math.pi/180*1800/math.pi)
    indexMajorLobePred = find_nearest(find_peaks(AFpred)[0], compDataX[k, 0]*math.pi/180*1800/math.pi)
    angleMajorLobeTrue = indexMajorLobeTrue*math.pi/1800
    angleMajorLobePred = indexMajorLobePred*math.pi/1800

    anglesNullsTrue = np.zeros(numOfSignals-1)
    anglesNullsPred = np.zeros(numOfSignals-1)
    for r in range(numOfSignals-1):
        anglesNullsTrue[r] = find_nearest(find_peaks(-AFtrue)[0], compDataX[k, r+1]*math.pi/180*1800/math.pi)*math.pi/1800
        anglesNullsPred[r] = find_nearest(find_peaks(-AFpred)[0], compDataX[k, r+1]*math.pi/180*1800/math.pi)*math.pi/1800

    angleMajorLobeTrue = angleMajorLobeTrue*180/math.pi
    angleMajorLobePred = angleMajorLobePred*180/math.pi
    anglesNullsTrue = anglesNullsTrue*180/math.pi
    anglesNullsPred = anglesNullsPred*180/math.pi

    diffDesired[k] = np.absolute(angleMajorLobeTrue-angleMajorLobePred)
    diffUndesired1[k] = np.absolute(anglesNullsTrue[0]-anglesNullsPred[0])
    diffUndesired2[k] = np.absolute(anglesNullsTrue[1]-anglesNullsPred[1])

print(np.mean(diffDesired))
print(np.mean(diffUndesired1))
print(np.mean(diffUndesired2))

#Plot losses in training and test sets
plt.figure()
plt.plot(trainLoss, color='b')
plt.plot(testLoss, color='r')

plt.figure()
plt.plot(trainLossPerBatch, color='b')

plt.figure()
plt.plot(diffDesired, color='b')

plt.figure()
plt.plot(diffUndesired1, color='b')

plt.figure()
plt.plot(diffUndesired2, color='b')
plt.show()

# What is new?
# A mini-batch of 20 has been set. The sum of losses per 20 examples is calculated and a single optimize step is taken using the mean loss of the mini-batch,
# instead of doing a step per example. This results in steady progress and reduces variance of each step due to variations in gradient per example (like in previous
# implementations)

# Problems of this attempt:
# 1) Did not train a lot (https://www.researchgate.net/post/What_is_the_minimum_sample_size_required_to_train_a_Deep_Learning_model-CNN)
#    
# 2) The input doesn't allow for much learning (a heatmap would be preferred rather than a 0-1 input image) (ok)
# 3) The learning rate may be too small to compensate the weights for the examples with great error 
#    and thus the loss functipn cannot be minimzed that much (big changes in weights may shfit but not reduce the loss) (ok)
# 4) The loss function may not be ideal: Better check difference between major lobe and nulls locations (ok)
# 5) The model requires some more nonlinearity (instead of leaky relu use sigmoid after conv layers) (?-stick to TinyYolo)

# Notes:
# 1) Mathematical Analysis of Loss Function: Does it have local minima, etc

# Corrections:
# 1) Apply random weights
# 2) Increase learning rate (ok)

# Next Steps:
# 1) Compare time complexity to current method (inversion of matrix)
# 2) Instead of the AF ideal raditaion pattern use the formula for radiation pattern of true antenna
