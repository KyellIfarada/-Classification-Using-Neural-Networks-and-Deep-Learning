### Only Numpy Library is necessary for this project and all the layers are compiled with Numpy.
import numpy as np 
import pickle 
import sys
import time
import pdb          ### library for debugging
from matplotlib import pyplot as plt 
 
np.random.seed(1000)

class Convolution2D:
    # Initialization of convolutional layer
    def __init__(self, inputs_channel, num_filters, kernel_size, padding, stride, learning_rate, name):
        # weight size: (F, C, K, K)
        # bias size: (F) 
        self.F = num_filters
        self.K = kernel_size
        self.C = inputs_channel

        self.weights = np.zeros((self.F, self.C, self.K, self.K))
        self.bias = np.zeros((self.F, 1))
        for i in range(0,self.F):
            self.weights[i,:,:,:] = np.random.normal(loc=0, scale=np.sqrt(1./(self.C*self.K*self.K)), size=(self.C, self.K, self.K))

        self.p = padding
        self.s = stride
        self.lr = learning_rate
        self.name = name
    
    # Padding Layer 
    def zero_padding(self, inputs, size):
        w, h = inputs.shape[0], inputs.shape[1]
        new_w = 2 * size + w
        new_h = 2 * size + h
        out = np.zeros((new_w, new_h))
        out[size:w+size, size:h+size] = inputs
        return out
    
    # Forward propagation
    def forward(self, inputs):
        # input size: (C, W, H)
        # output size: (N, F ,WW, HH)
        C = inputs.shape[0]
        W = inputs.shape[1]+2*self.p
        H = inputs.shape[2]+2*self.p
        self.inputs = np.zeros((C, W, H))
        for c in range(inputs.shape[0]):
            self.inputs[c,:,:] = self.zero_padding(inputs[c,:,:], self.p)
        WW = (W - self.K)//self.s + 1
        HH = (H - self.K)//self.s + 1
        feature_maps = np.zeros((self.F, WW, HH))
        for f in range(self.F):
            for w in range(WW):
                for h in range(HH):
                    feature_maps[f,w,h]=np.sum(self.inputs[:,w:w+self.K,h:h+self.K]*self.weights[f,:,:,:])+self.bias[f]

        return feature_maps
    
    # Backward Propagation
    def backward(self, dy):

        C, W, H = self.inputs.shape
        dx = np.zeros(self.inputs.shape)
        dw = np.zeros(self.weights.shape)
        db = np.zeros(self.bias.shape)

        F, W, H = dy.shape
        for f in range(F):
            for w in range(W):
                for h in range(H):
                    dw[f,:,:,:]+=dy[f,w,h]*self.inputs[:,w:w+self.K,h:h+self.K]
                    dx[:,w:w+self.K,h:h+self.K]+=dy[f,w,h]*self.weights[f,:,:,:]

        for f in range(F):
            db[f] = np.sum(dy[f, :, :])

        self.weights -= self.lr * dw
        self.bias -= self.lr * db
        return dx
    
    # Function for extract the weights and bias for storage
    def extract(self):
        return {self.name+'.weights':self.weights, self.name+'.bias':self.bias}
    
    # Feed the pretrained weights and bias for models 
    def feed(self, weights, bias):
        self.weights = weights
        self.bias = bias

class Maxpooling2D:
    # Initialization of MaxPooling layer
    def __init__(self, pool_size, stride, name):
        self.pool = pool_size
        self.s = stride
        self.name = name
    
    # Forward propagation
    def forward(self, inputs):
        self.inputs = inputs
        C, W, H = inputs.shape
        new_width = (W - self.pool)//self.s + 1
        new_height = (H - self.pool)//self.s + 1
        out = np.zeros((C, new_width, new_height))
        for c in range(C):
            for w in range(W//self.s):
                for h in range(H//self.s):
                    out[c, w, h] = np.max(self.inputs[c, w*self.s:w*self.s+self.pool, h*self.s:h*self.s+self.pool])
        return out
    
    # Backward propagation
    def backward(self, dy):
        C, W, H = self.inputs.shape
        dx = np.zeros(self.inputs.shape)
        
        for c in range(C):
            for w in range(0, W, self.pool):
                for h in range(0, H, self.pool):
                    st = np.argmax(self.inputs[c,w:w+self.pool,h:h+self.pool])
                    (idx, idy) = np.unravel_index(st, (self.pool, self.pool))
                    dx[c, w+idx, h+idy] = dy[c, w//self.pool, h//self.pool]
        return dx
    
    # No weights and bias for pooling layer to store
    def extract(self):
        return 

class FullyConnected:
    # Initialization of Fully-Connected Layer
    def __init__(self, num_inputs, num_outputs, learning_rate, name):
        self.weights = 0.01*np.random.rand(num_inputs, num_outputs)
        self.bias = np.zeros((num_outputs, 1))
        self.lr = learning_rate
        self.name = name
    
    # Forward Propagation
    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(self.inputs, self.weights) + self.bias.T
    
    # Backward Propagation
    def backward(self, dy):

        if dy.shape[0] == self.inputs.shape[0]:
            dy = dy.T
        dw = dy.dot(self.inputs)
        db = np.sum(dy, axis=1, keepdims=True)
        dx = np.dot(dy.T, self.weights.T)

        self.weights -= self.lr * dw.T
        self.bias -= self.lr * db

        return dx
    
    # Extract weights and bias for storage
    def extract(self):
        return {self.name+'.weights':self.weights, self.name+'.bias':self.bias}
    
    # Feed the pretrained weights and bias for models 
    def feed(self, weights, bias):
        self.weights = weights
        self.bias = bias

### Flatten function to convert 4D feature maps into 3D feature vectors
class Flatten:
    def __init__(self):
        pass
    def forward(self, inputs):
        self.C, self.W, self.H = inputs.shape
        return inputs.reshape(1, self.C*self.W*self.H)
    def backward(self, dy):
        return dy.reshape(self.C, self.W, self.H)
    def extract(self):
        return
		
### ReLU activation function
class ReLu:
    def __init__(self):
        pass
    def forward(self, inputs):
        self.inputs = inputs
        ret = inputs.copy()
        ret[ret < 0] = 0
        return ret
    def backward(self, dy):
        dx = dy.copy()
        dx[self.inputs < 0] = 0
        return dx
    def extract(self):
        return

### Softmax activation function
class Softmax:
    def __init__(self):
        pass
    def forward(self, inputs):
        exp = np.exp(inputs, dtype=np.float)
        self.out = exp/np.sum(exp)
        return self.out
    def backward(self, dy):
        return self.out.T - dy.reshape(dy.shape[0],1)
    def extract(self):
        return
		
from precode import *
"""
The subset of MNIST is created based on the last 4 digits of your ASUID. There are 4 categories and all returned 
samples are preprocessed and shuffled. 
"""
print('Loading data......')
sub_train_images, sub_train_labels, sub_test_images, sub_test_labels = init_subset('9328') # input your ASUID last 4 digits here to generate the subset samples of MNIST for training and testing		

net = Net()
epoch = 10            ### Default number of epochs
batch_size = 100      ### Default batch size
num_batch = sub_train_images.shape[0]/batch_size
test_accuracyGraph=np.zeros((10,1))
train_accuracyGraph=np.zeros((10,1))
test_lossGraph=np.zeros((10,1))
train_lossGraph=np.zeros((10,1))
epoch_Graph = np.array((1,2,3,4,5,6,7,8,9,10))
test_size = sub_test_images.shape[0]      # Obtain the size of testing samples
train_size = sub_train_images.shape[0]    # Obtain the size of training samples
### ---------- ###
"""
Please compile your own evaluation code based on the training code 
to evaluate the trained network.
The function name and the inputs of the function have been predifined and please finish the remaining part.
"""
def evaluate(net, images, labels):
# baseline constants    
    acc = 0    
    loss = 0
    batch_size = 1
    sample_size =0
#Choose Sample_Size to Normalize Accuracy/Loss
    if images.shape[0] == test_size:
        sample_size = test_size
    if images.shape[0] == train_size:
        sample_size = train_size
#Use For Counter to increment by 1 and iter within sample pool by 1     
    for batch_index in range(0, images.shape[0], batch_size):     
            x = images[batch_index]
            y = labels[batch_index]
    
# Pass Samples By 1 to perform CNN 
            for l in range(net.lay_num):
                output = net.layers[l].forward(x)
                x = output
#Calculate Loss /Accuracy         
            loss += cross_entropy(output, y)
            if np.argmax(output) == np.argmax(y):
                acc += 1
#Normalize Results by Sample_Size for percentage basis         
    loss /= (sample_size)
    acc  /= (sample_size)
         
    return float(acc), float(loss)

### Start training process
for e in range(epoch):
    total_acc = 0    
    total_loss = 0
    print('Epoch %d' % e)
    for batch_index in range(0, sub_train_images.shape[0], batch_size):
        # batch input
        if batch_index + batch_size < sub_train_images.shape[0]:
            data = sub_train_images[batch_index:batch_index+batch_size]
            label = sub_train_labels[batch_index:batch_index + batch_size]
        else:
            data = sub_train_images[batch_index:sub_train_images.shape[0]]
            label = sub_train_labels[batch_index:sub_train_labels.shape[0]]
        # Compute the remaining time
        start_time = time.time()
        batch_loss,batch_acc = net.train(data, label)  # Train the network with samples in one batch 
        
        end_time = time.time()
        batch_time = end_time-start_time
        remain_time = (sub_train_images.shape[0]-batch_index)/batch_size*batch_time
        hrs = int(remain_time/3600)
        mins = int((remain_time/60-hrs*60))
        secs = int(remain_time-mins*60-hrs*3600)
        print('=== Iter:{0:d} === Remain: {1:d} Hrs {2:d} Mins {3:d} Secs ==='.format(int(batch_index+batch_size),int(hrs),int(mins),int(secs)))
    
    # Print out the Performance

    train_acc, train_loss = evaluate(net, sub_train_images, sub_train_labels)  # Use the evaluation code to obtain the training accuracy and loss
    test_acc, test_loss = evaluate(net, sub_test_images, sub_test_labels)      # Use the evaluation code to obtain the testing accuracy and loss
    test_accuracyGraph[e] = test_acc
    train_accuracyGraph[e] = train_acc
    test_lossGraph[e] = test_loss
    train_lossGraph[e] = train_loss
    print('=== Epoch:{0:d} Train Size:{1:d}, Train Acc:{2:.3f}, Train Loss:{3:.3f} ==='.format(e, train_size,train_acc,train_loss))
    print('=== Epoch:{0:d} Test Size:{1:d}, Test Acc:{2:.3f}, Test Loss:{3:.3f} ==='.format(e, test_size, test_acc,test_loss))
    
plt.title("Epoch v TrainAcc") 
plt.xlabel("x axis epoch") 
plt.ylabel("y axis train_acc") 
plt.plot(epoch_Graph,train_accuracyGraph) 
plt.show()

plt.title("Epoch v TrainLoss")
plt.xlabel("x axis epoch") 
plt.ylabel("y train_loss ") 
plt.plot(epoch_Graph,train_lossGraph) 
plt.show()

plt.title("Epoch v TestAcc")
plt.xlabel("x axis epoch") 
plt.ylabel("y axis test_accuracy") 
plt.plot(epoch_Graph,test_accuracyGraph) 
plt.show()

plt.title("Epoch v TrainLoss") 
plt.xlabel("x axis epoch") 
plt.ylabel("y axis test_loss") 
plt.plot(epoch_Graph,test_lossGraph) 
plt.show()

