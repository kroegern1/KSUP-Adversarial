
# coding: utf-8

# In[1]:

# get_ipython().magic(u'matplotlib inline')
import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import metrics
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize



# In[2]:

dev = mx.cpu()
batch_size = 100
data_shape = (1, 28, 28)

train_iter = mx.io.MNISTIter(
        image       = "../softwares/Thesis/adversarial/LearningwithaStrongAdversary-master/datasets/mnist/train-images.idx3-ubyte",
        label       = "../softwares/Thesis/adversarial/LearningwithaStrongAdversary-master/datasets/mnist/train-labels.idx1-ubyte",
        input_shape = data_shape,
        batch_size  = batch_size,
        shuffle     = True,
        flat        = False)

val_iter = mx.io.MNISTIter(
        image       = "../softwares/Thesis/adversarial/LearningwithaStrongAdversary-master/datasets/mnist/t10k-images.idx3-ubyte",
        label       = "../softwares/Thesis/adversarial/LearningwithaStrongAdversary-master/datasets/mnist/t10k-labels.idx1-ubyte",
        input_shape = data_shape,
        batch_size  = batch_size,
        flat        = False)


# In[3]:

def Softmax(theta):
    max_val = np.max(theta, axis=1, keepdims=True)
    tmp = theta - max_val
    exp = np.exp(tmp)
    norm = np.sum(exp, axis=1, keepdims=True)
    return exp / norm

    
def LogLossGrad(alpha, label):
    grad = np.copy(alpha)
    for i in range(alpha.shape[0]):
        grad[i, label[i]] -= 1.
    return grad


def SGD(weight, grad, lr=0.1, grad_norm=batch_size):
    weight[:] -= lr * grad / batch_size


def CalAcc(pred_prob, label):
    pred = np.argmax(pred_prob, axis=1)
    return np.sum(pred == label) * 1.0


def CalLoss(pred_prob, label):
    loss = 0.
    for i in range(pred_prob.shape[0]):
        loss += -np.log(max(pred_prob[i, label[i]], 1e-10))
    return loss




# # Normal Training

# In[ ]:

# input
data = mx.symbol.Variable('data')
# first fullc
flatten = mx.symbol.Flatten(data=data)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=100)
relu1 = mx.symbol.Activation(data=fc1, act_type="relu")
fc2 = mx.symbol.FullyConnected(data=relu1, num_hidden=100)
relu2 = mx.symbol.Activation(data=fc2, act_type="relu")

# second fullc
fc3 = mx.symbol.FullyConnected(data=relu2, num_hidden=10)


# In[ ]:

data_shape = (batch_size, 1, 28, 28)
arg_names = fc3.list_arguments() # 'data' 
arg_shapes, output_shapes, aux_shapes = fc3.infer_shape(data=data_shape)

arg_arrays = [mx.nd.zeros(shape, ctx=dev) for shape in arg_shapes]
grad_arrays = [mx.nd.zeros(shape, ctx=dev) for shape in arg_shapes]
reqs = ["write" for name in arg_names]

model = fc3.bind(ctx=dev, args=arg_arrays, args_grad = grad_arrays, grad_req=reqs)
arg_map = dict(zip(arg_names, arg_arrays))
grad_map = dict(zip(arg_names, grad_arrays))
data_grad = grad_map["data"]
out_grad = mx.nd.zeros(model.outputs[0].shape, ctx=dev)


# In[ ]:

for name in arg_names:
    if "weight" in name:
        arr = arg_map[name]
        arr[:] = mx.rnd.uniform(-0.07, 0.07, arr.shape)


# In[ ]:

num_round = 1
train_acc = 0.
nbatch = 0
coe_pb = 1.5
for i in range(num_round):
    train_loss = 0.
    train_acc = 0.
    nbatch = 0
    train_iter.reset()
    for dbatch in train_iter:
        data = dbatch.data[0]
        label = dbatch.label[0]
        arg_map["data"][:] = data
        model.forward(is_train=True)
        theta = model.outputs[0].asnumpy()
        alpha = Softmax(theta)
        train_acc += CalAcc(alpha, label.asnumpy()) / batch_size
        train_loss += CalLoss(alpha, label.asnumpy()) / batch_size
        losGrad_theta = LogLossGrad(alpha, label.asnumpy())
        out_grad[:] = losGrad_theta
        model.backward([out_grad])
        for name in arg_names:
            if name != "data":
                SGD(arg_map[name], grad_map[name])
        
        nbatch += 1
    train_acc /= nbatch
    train_loss /= nbatch
    #val_acc = acc_normal(model, val_iter,arg_map, grad_map)
    #print("Train Accuracy: %.4f\t Val Accuracy: %.4f\t Train Loss: %.5f" % (train_acc, val_acc, train_loss))


# In[ ]:


val_iter.reset()
val_acc = 0.0
num_batch = 0
count=1
for dbatch in val_iter:
        data = dbatch.data[0]
        label = dbatch.label[0]
        batch_size = label.asnumpy().shape[0]
        arg_map["data"][:] = data    

        model.forward(is_train=True)
        theta = model.outputs[0].asnumpy()
        alpha = Softmax(theta)
        
        grad = LogLossGrad(alpha, label.asnumpy())
        out_grad[:] = grad
        model.backward([out_grad])
        noise = grad_map["data"].asnumpy()
        
        for j in range(batch_size):
            if np.linalg.norm(noise[j].flatten(),2) ==0:
                nn+=1
            y = label.asnumpy()[j]
            if (y == np.argmax(alpha[j])):
                noise[j] = noise[j]/np.linalg.norm(noise[j].flatten(),2)
            else:
                noise[j] = 0
        pdata = data.asnumpy() + coe_pb * noise
        arg_map["data"][:] = pdata
        model.forward(is_train=False)
        raw_output = model.outputs[0].asnumpy()
        pred = Softmax(raw_output)
	if(count==1): 
    	    ypred=raw_output
            ytrue=label.asnumpy() 		
 	else:
     	    ypred=np.vstack((ypred,raw_output))
            ytrue=np.hstack((ytrue,label.asnumpy()))		
	count=count+1	



colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
lw = 2
ytrue=label_binarize(ytrue, classes=[0, 1, 2,3,4,5,6,7,8,9]) 
precision = dict()
recall = dict()
average_precision = dict()
precision["micro"], recall["micro"], _ = precision_recall_curve(ytrue.ravel(),ypred.ravel())
average_precision["micro"] = average_precision_score(ytrue, ypred,average="micro")
plt.clf()
plt.plot(recall["micro"], precision["micro"], color='gold', lw=lw,
label='Normal Training (area = {0:0.2f})'.format(average_precision["micro"]))



# # Dropout Training

# In[ ]:

# input
data = mx.symbol.Variable('data')
# first fullc
flatten = mx.symbol.Flatten(data=data)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=100)
relu1 = mx.symbol.Activation(data=fc1, act_type="relu")
fc2 = mx.symbol.FullyConnected(data=relu1, num_hidden=100)
relu2 = mx.symbol.Activation(data=fc2, act_type="relu")
dropout1 = mx.symbol.Dropout(data=relu2, p=0.5)
# second fullc
fc3 = mx.symbol.FullyConnected(data=dropout1, num_hidden=10)


# In[ ]:

data_shape = (batch_size, 1, 28, 28)
arg_names = fc3.list_arguments() # 'data' 
arg_shapes, output_shapes, aux_shapes = fc3.infer_shape(data=data_shape)

arg_arrays = [mx.nd.zeros(shape, ctx=dev) for shape in arg_shapes]
grad_arrays = [mx.nd.zeros(shape, ctx=dev) for shape in arg_shapes]
reqs = ["write" for name in arg_names]

model = fc3.bind(ctx=dev, args=arg_arrays, args_grad = grad_arrays, grad_req=reqs)
arg_map = dict(zip(arg_names, arg_arrays))
grad_map = dict(zip(arg_names, grad_arrays))
data_grad = grad_map["data"]
out_grad = mx.nd.zeros(model.outputs[0].shape, ctx=dev)


# In[ ]:

for name in arg_names:
    if "weight" in name:
        arr = arg_map[name]
        arr[:] = mx.rnd.uniform(-0.07, 0.07, arr.shape)

# In[ ]:

num_round = 1
train_acc = 0.
nbatch = 0
lr = 0.01
coe_pb = 1.5
for i in range(num_round):
    train_loss = 0.
    train_acc = 0.
    nbatch = 0
    train_iter.reset()
    for dbatch in train_iter:
        data = dbatch.data[0]
        label = dbatch.label[0]
        arg_map["data"][:] = data
        model.forward(is_train=True)
        theta = model.outputs[0].asnumpy()
        alpha = Softmax(theta)
        train_acc += CalAcc(alpha, label.asnumpy()) / batch_size
        train_loss += CalLoss(alpha, label.asnumpy()) / batch_size
        losGrad_theta = LogLossGrad(alpha, label.asnumpy())
        out_grad[:] = losGrad_theta
        model.backward([out_grad])
        for name in arg_names:
            if name != "data":
                SGD(arg_map[name], grad_map[name],lr)
        
        nbatch += 1
    train_acc /= nbatch
    train_loss /= nbatch
    #val_acc = acc_normal(model, val_iter,arg_map, grad_map)
    #print("Train Accuracy: %.4f\t Val Accuracy: %.4f\t Train Loss: %.5f" % (train_acc, val_acc, train_loss))


# In[ ]:

val_iter.reset()
val_acc = 0.0
num_batch = 0
count=1
for dbatch in val_iter:
        data = dbatch.data[0]
        label = dbatch.label[0]
        batch_size = label.asnumpy().shape[0]
        arg_map["data"][:] = data    

        model.forward(is_train=True)
        theta = model.outputs[0].asnumpy()
        alpha = Softmax(theta)
        
        grad = LogLossGrad(alpha, label.asnumpy())
        out_grad[:] = grad
        model.backward([out_grad])
        noise = grad_map["data"].asnumpy()
        
        for j in range(batch_size):
            if np.linalg.norm(noise[j].flatten(),2) ==0:
                nn+=1
            y = label.asnumpy()[j]
            if (y == np.argmax(alpha[j])):
                noise[j] = noise[j]/np.linalg.norm(noise[j].flatten(),2)
            else:
                noise[j] = 0
        pdata = data.asnumpy() + coe_pb * noise
        arg_map["data"][:] = pdata
        model.forward(is_train=False)
        raw_output = model.outputs[0].asnumpy()
        pred = Softmax(raw_output)
	if(count==1): 
    	    ypred=raw_output
            ytrue=label.asnumpy() 		
 	else:
     	    ypred=np.vstack((ypred,raw_output))
            ytrue=np.hstack((ytrue,label.asnumpy()))		
	count=count+1	



lw = 2
ytrue=label_binarize(ytrue, classes=[0, 1, 2,3,4,5,6,7,8,9]) 
precision3 = dict()
recall3 = dict()
average_precision3 = dict()
precision3["micro"], recall3["micro"], _ = precision_recall_curve(ytrue.ravel(),ypred.ravel())
average_precision3["micro"] = average_precision_score(ytrue, ypred,average="micro")
plt.plot(recall3["micro"], precision3["micro"], color='darkorange', lw=lw,
label='Dropout Training (area = {0:0.2f})'.format(average_precision3["micro"]))


# # Ian's Method

# In[ ]:

# input
data = mx.symbol.Variable('data')
# first fullc
flatten = mx.symbol.Flatten(data=data)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=100)
relu1 = mx.symbol.Activation(data=fc1, act_type="relu")
fc2 = mx.symbol.FullyConnected(data=relu1, num_hidden=100)
relu2 = mx.symbol.Activation(data=fc2, act_type="relu")
# second fullc
fc3 = mx.symbol.FullyConnected(data=relu2, num_hidden=10)


# In[ ]:

data_shape = (batch_size, 1, 28, 28)
arg_names = fc3.list_arguments() # 'data' 
arg_shapes, output_shapes, aux_shapes = fc3.infer_shape(data=data_shape)

arg_arrays = [mx.nd.zeros(shape, ctx=dev) for shape in arg_shapes]
grad_arrays = [mx.nd.zeros(shape, ctx=dev) for shape in arg_shapes]
sum_arrays = [mx.nd.zeros(shape, ctx=dev) for shape in arg_shapes]

reqs = ["write" for name in arg_names]

model = fc3.bind(ctx=dev, args=arg_arrays, args_grad = grad_arrays, grad_req=reqs)
arg_map = dict(zip(arg_names, arg_arrays))
grad_map = dict(zip(arg_names, grad_arrays))
sum_map = dict(zip(arg_names, sum_arrays))
data_grad = grad_map["data"]
out_grad = mx.nd.zeros(model.outputs[0].shape, ctx=dev)


# In[ ]:

for name in arg_names:
    if "weight" in name:
        arr = arg_map[name]
        arr[:] = mx.rnd.uniform(-0.07, 0.07, arr.shape)
for name in arg_names:
    sum_map[name][:] = 0.




# In[ ]:

num_round = 1
train_acc = 0.
nbatch = 0
coe_pb = 1.5
lr= 0.005
for i in range(num_round):
    train_loss = 0.
    train_acc = 0.
    nbatch = 0
    train_iter.reset()
    for dbatch in train_iter:
        data = dbatch.data[0]
        label = dbatch.label[0]
        arg_map["data"][:] = data
        model.forward(is_train=True)
        theta = model.outputs[0].asnumpy()
        alpha = Softmax(theta)
        train_acc += CalAcc(alpha, label.asnumpy()) / batch_size
        train_loss += CalLoss(alpha, label.asnumpy()) / batch_size
        losGrad_theta = LogLossGrad(alpha, label.asnumpy())
        out_grad[:] = losGrad_theta
        model.backward([out_grad])
        
        for name in arg_names:
            if name != "data":
                sum_map[name][:] += grad_map[name]
        #grad1 = grad_map
        
        noise = np.sign(data_grad.asnumpy())
        for j in range(batch_size):
            noise[j] = noise[j]/np.linalg.norm(noise[j].flatten(),2)
        pdata = data.asnumpy() + coe_pb * noise
        arg_map["data"][:] = pdata
        model.forward(is_train=True)
        theta = model.outputs[0].asnumpy()
        alpha = Softmax(theta)
        losGrad_theta = LogLossGrad(alpha, label.asnumpy())
        out_grad[:] = losGrad_theta
        model.backward([out_grad])
        
        for name in arg_names:
            if name != "data":
                sum_map[name][:] += grad_map[name]

        for name in arg_names:
            if name != "data":
                SGD(arg_map[name], sum_map[name], lr)
            sum_map[name][:] = 0.
        
        nbatch += 1
    train_acc /= nbatch
    train_loss /= nbatch
    #val_acc = acc_normal(model, val_iter,arg_map, grad_map)
    #print("Train Accuracy: %.4f\t Val Accuracy: %.4f\t Train Loss: %.5f" % (train_acc, val_acc, train_loss))


# In[ ]:

val_iter.reset()
val_acc = 0.0
num_batch = 0
count=1
for dbatch in val_iter:
        data = dbatch.data[0]
        label = dbatch.label[0]
        batch_size = label.asnumpy().shape[0]
        arg_map["data"][:] = data    

        model.forward(is_train=True)
        theta = model.outputs[0].asnumpy()
        alpha = Softmax(theta)
        
        grad = LogLossGrad(alpha, label.asnumpy())
        out_grad[:] = grad
        model.backward([out_grad])
        noise = grad_map["data"].asnumpy()
        
        for j in range(batch_size):
            if np.linalg.norm(noise[j].flatten(),2) ==0:
                nn+=1
            y = label.asnumpy()[j]
            if (y == np.argmax(alpha[j])):
                noise[j] = noise[j]/np.linalg.norm(noise[j].flatten(),2)
            else:
                noise[j] = 0
        pdata = data.asnumpy() + coe_pb * noise
        arg_map["data"][:] = pdata
        model.forward(is_train=False)
        raw_output = model.outputs[0].asnumpy()
        pred = Softmax(raw_output)
	if(count==1): 
    	    ypred=raw_output
            ytrue=label.asnumpy() 		
 	else:
     	    ypred=np.vstack((ypred,raw_output))
            ytrue=np.hstack((ytrue,label.asnumpy()))		
	count=count+1	


	


lw = 2
ytrue=label_binarize(ytrue, classes=[0, 1, 2,3,4,5,6,7,8,9]) 
precision4 = dict()
recall4 = dict()
average_precision4 = dict()
precision4["micro"], recall4["micro"], _ = precision_recall_curve(ytrue.ravel(),ypred.ravel())
average_precision4["micro"] = average_precision_score(ytrue, ypred,average="micro")
plt.plot(recall4["micro"], precision4["micro"], color='turquoise', lw=lw,
label="Goodfellow's Training (area = {0:0.2f})".format(average_precision4["micro"]))


# # Ksup + dropout

# In[ ]:

# input
data = mx.symbol.Variable('data')
# first fullc
flatten = mx.symbol.Flatten(data=data)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=100)
relu1 = mx.symbol.Activation(data=fc1, act_type="relu")
fc2 = mx.symbol.FullyConnected(data=relu1, num_hidden=100)
relu2 = mx.symbol.Activation(data=fc2, act_type="relu")
dropout1 = mx.symbol.Dropout(data=relu2, p=0.5)
# second fullc
fc3 = mx.symbol.FullyConnected(data=dropout1, num_hidden=10)


# In[ ]:

data_shape = (batch_size, 1, 28, 28)
arg_names = fc3.list_arguments() # 'data' 
arg_shapes, output_shapes, aux_shapes = fc3.infer_shape(data=data_shape)

arg_arrays = [mx.nd.zeros(shape, ctx=dev) for shape in arg_shapes]
grad_arrays = [mx.nd.zeros(shape, ctx=dev) for shape in arg_shapes]
reqs = ["write" for name in arg_names]

model = fc3.bind(ctx=dev, args=arg_arrays, args_grad = grad_arrays, grad_req=reqs)
arg_map = dict(zip(arg_names, arg_arrays))
grad_map = dict(zip(arg_names, grad_arrays))
data_grad = grad_map["data"]
out_grad = mx.nd.zeros(model.outputs[0].shape, ctx=dev)


# In[ ]:

for name in arg_names:
    if "weight" in name:
        arr = arg_map[name]
        arr[:] = mx.rnd.uniform(-0.07, 0.07, arr.shape)


# In[ ]:

num_round = 1
train_acc = 0.
nbatch = 0
coe_pb = 1.5
lr = 0.1
for i in range(num_round):
    train_loss = 0.
    train_acc = 0.
    nbatch = 0
    train_iter.reset()
    for dbatch in train_iter:
        data = dbatch.data[0]
        label = dbatch.label[0]
        arg_map["data"][:] = data
        model.forward(is_train=True)
        theta = model.outputs[0].asnumpy()
        alpha = Softmax(theta)
        train_acc += CalAcc(alpha, label.asnumpy()) / batch_size
        train_loss += CalLoss(alpha, label.asnumpy()) / batch_size
        losGrad_theta = LogLossGrad(alpha, label.asnumpy())
        out_grad[:] = losGrad_theta
        model.backward([out_grad])
	new_grad= data_grad.asnumpy()
        gresh=np.zeros((len(new_grad),784),dtype=float)
        for i in np.arange(len(new_grad)):
            temp_row=np.squeeze(new_grad[i])
            gresh[i]=temp_row.flatten()
            
        n=int(np.floor(.50*784))
        asorted=np.argsort(np.absolute(gresh),axis=-1, kind='quicksort')
        mask=np.zeros((gresh.shape),dtype=float)
        for i in np.arange(len(asorted)):
            c=(asorted[i][-n:]).tolist()
            for ind,val in enumerate(c):
                 mask[i,val]=1.0
                 
            
        temp_noise=np.multiply(gresh,mask)
        noise=np.reshape(temp_noise, (new_grad.shape))
        for j in range(batch_size):
            noise[j] = noise[j]/np.linalg.norm(noise[j].flatten(),2)
        pdata = data.asnumpy() + coe_pb * noise
        arg_map["data"][:] = pdata
        model.forward(is_train=True)
        theta = model.outputs[0].asnumpy()
        alpha = Softmax(theta)
        losGrad_theta = LogLossGrad(alpha, label.asnumpy())
        out_grad[:] = losGrad_theta
        model.backward([out_grad])
        for name in arg_names:
            if name != "data":
                SGD(arg_map[name], grad_map[name], lr)
        
        nbatch += 1
    train_acc /= nbatch
    train_loss /= nbatch
    #val_acc = acc_normal(model, val_iter,arg_map, grad_map)
    #print("Train Accuracy: %.4f\t Val Accuracy: %.4f\t Train Loss: %.5f" % (train_acc, val_acc, train_loss))



# In[ ]:

val_iter.reset()
val_acc = 0.0
num_batch = 0
count=1
for dbatch in val_iter:
        data = dbatch.data[0]
        label = dbatch.label[0]
        batch_size = label.asnumpy().shape[0]
        arg_map["data"][:] = data    

        model.forward(is_train=True)
        theta = model.outputs[0].asnumpy()
        alpha = Softmax(theta)
        
        grad = LogLossGrad(alpha, label.asnumpy())
        out_grad[:] = grad
        model.backward([out_grad])
        noise = grad_map["data"].asnumpy()
        
        for j in range(batch_size):
            if np.linalg.norm(noise[j].flatten(),2) ==0:
                nn+=1
            y = label.asnumpy()[j]
            if (y == np.argmax(alpha[j])):
                noise[j] = noise[j]/np.linalg.norm(noise[j].flatten(),2)
            else:
                noise[j] = 0
        pdata = data.asnumpy() + coe_pb * noise
        arg_map["data"][:] = pdata
        model.forward(is_train=False)
        raw_output = model.outputs[0].asnumpy()
        pred = Softmax(raw_output)
	if(count==1): 
    	    ypred=raw_output
            ytrue=label.asnumpy() 		
 	else:
     	    ypred=np.vstack((ypred,raw_output))
            ytrue=np.hstack((ytrue,label.asnumpy()))		
	count=count+1	


	


colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
lw = 2
ytrue=label_binarize(ytrue, classes=[0, 1, 2,3,4,5,6,7,8,9]) 
precision2 = dict()
recall2 = dict()
average_precision2 = dict()
precision2["micro"], recall2["micro"], _ = precision_recall_curve(ytrue.ravel(),ypred.ravel())
average_precision2["micro"] = average_precision_score(ytrue, ypred,average="micro")
#plt.clf()
plt.plot(recall2["micro"], precision2["micro"], color='navy', lw=lw,
label='KSUP Training (area = {0:0.2f})'.format(average_precision2["micro"]))
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curves using L2 noised Test Data')
plt.legend(loc="upper right")
plt.show()




