
# coding: utf-8

# In[1]:


# get_ipython().magic(u'matplotlib inline')
import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# In[2]:

dev = mx.cpu()
batch_size = 100
data_shape = (1, 28, 28)

train_iter = mx.io.MNISTIter(
        image       = "../softwares/adversarial/LearningwithaStrongAdversary-master/datasets/mnist/train-images.idx3-ubyte",
        label       = "../softwares/adversarial/LearningwithaStrongAdversary-master/datasets/mnist/train-labels.idx1-ubyte",
        input_shape = data_shape,
        batch_size  = batch_size,
        shuffle     = True,
        flat        = False)

val_iter = mx.io.MNISTIter(
        image       = "../softwares/adversarial/LearningwithaStrongAdversary-master/datasets/mnist/t10k-images.idx3-ubyte",
        label       = "../softwares/adversarial/LearningwithaStrongAdversary-master/datasets/mnist/t10k-labels.idx1-ubyte",
        input_shape = data_shape,
        batch_size  = batch_size,
        flat        = False)

# In[5]:

def Softmax(theta):
    max_val = np.max(theta, axis=1, keepdims=True)
    tmp = theta - max_val
    exp = np.exp(tmp)
    norm = np.sum(exp, axis=1, keepdims=True)
    return exp / norm

def MultiHinge(theta, label):
    tmp1 = theta-theta[label]
    tmp1[label] = -np.inf
    tmp2 = np.max(tmp1)+1
    return np.max(tmp2, 0)
    
def MultiHingeGrad(theta, label):
    tmp = np.zeros(theta.shape)
    tmp1 = theta-theta[label]
    tmp1[label] = -np.inf
    if np.max(tmp1)+1 < 0:
        return tmp
    else:
        ind = np.argmax(tmp1)
        tmp[ind] = 1
        tmp[label] = -1
        return tmp
    
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


# In[6]:

def acc_normal(model, val_iter, arg_map, grad_map):
    val_iter.reset()
    val_acc = 0.0
    num_samp = 0
    for dbatch in val_iter:
        data = dbatch.data[0]
        label = dbatch.label[0]
        batch_size = label.asnumpy().shape[0]
        arg_map["data"][:] = data    

        model.forward(is_train=False)
        theta = model.outputs[0].asnumpy()
        alpha = Softmax(theta)
        val_acc += CalAcc(alpha, label.asnumpy()) 
        num_samp += batch_size
    return(val_acc / num_samp)
    
def acc_perb_L0(model, val_iter, coe_pb,arg_map, grad_map):
    val_iter.reset()
    val_acc = 0.0
    num_samp = 0
    nn=0
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
        noise = np.sign(grad_map["data"].asnumpy())
        
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
        
        val_acc += CalAcc(pred, label.asnumpy()) 
        num_samp += batch_size
    if  nn>0:
        print('L0 gradien being 0 :', nn)
    return(val_acc / num_samp)

def acc_perb_L2(model, val_iter, coe_pb, arg_map, grad_map):
    val_iter.reset()
    val_acc = 0.0
    num_batch = 0
    nn=0
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
        
        val_acc += CalAcc(pred, label.asnumpy()) /  batch_size 
        num_batch += 1
    if  nn>0:
        print('L2 gradien being 0 :', nn)
    return(val_acc / num_batch)


def acc_perb_ks(model,n, val_iter, coe_pb, arg_map, grad_map):
    val_iter.reset()
    val_acc = 0.0
    num_batch = 0
    nn=0
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
        new_grad= grad_map["data"].asnumpy()
        gresh=np.zeros((len(new_grad),784),dtype=float)
        for i in np.arange(len(new_grad)):
            temp_row=np.squeeze(new_grad[i])
            gresh[i]=temp_row.flatten()
            
        #n=int(np.floor(.95*784))
        asorted=np.argsort(np.absolute(gresh),axis=-1, kind='quicksort')
        mask=np.zeros((gresh.shape),dtype=float)
        for i in np.arange(len(asorted)):
            c=(asorted[i][-n:]).tolist()
            for ind,val in enumerate(c):
                 mask[i,val]=1.0
                 
            
        temp_noise=np.multiply(gresh,mask)
        noise=np.reshape(temp_noise, (new_grad.shape))
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
        val_acc += CalAcc(pred, label.asnumpy()) /  batch_size 
        num_batch += 1
    if  nn>0:
        print('ks gradien being 0 :', nn)
    return(val_acc / num_batch)

def acc_perb_alpha(model, val_iter, coe_pb,arg_map, grad_map):
    val_iter.reset()
    val_acc = 0.0
    num_samp = 0
    nn=0
    for dbatch in val_iter:
        data = dbatch.data[0]
        label = dbatch.label[0]
        batch_size = label.asnumpy().shape[0]
        arg_map["data"][:] = data    

        T = np.zeros((10, batch_size, data_shape[1], data_shape[2], data_shape[3]))
        noise = np.zeros(data.shape)
        #===================
        for i in range(10):
            arg_map["data"][:] = data   
            model.forward(is_train=True)
            theta = model.outputs[0].asnumpy()
            alpha = Softmax(theta)
            
            grad = LogLossGrad(alpha, i*np.ones(alpha.shape[0]))
            for j in range(batch_size):
                grad[j] = -alpha[j][i]*grad[j]
            out_grad[:] = grad
            model.backward([out_grad])
            T[i] = grad_map["data"].asnumpy()
        
        for j in range(batch_size):
            y = label.asnumpy()[j]
            if (y == np.argmax(alpha[j])): 
                perb_scale = np.zeros(10)
                for i in range(10):
                    if (i == y):
                        perb_scale[i] = np.inf
                    else:
                        perb_scale[i] = (alpha[j][y] - alpha[j][i])/np.linalg.norm((T[i][j]-T[y][j]).flatten(),2)
                noise[j] = T[np.argmin(perb_scale)][j]-T[y][j]
        #====================
        for j in range(batch_size):
            if np.linalg.norm(noise[j].flatten(),2) ==0:
                nn+=1
            else:
                noise[j] = noise[j]/np.linalg.norm(noise[j].flatten(),2)
        pdata = data.asnumpy() + coe_pb * noise
        arg_map["data"][:] = pdata
        model.forward(is_train=False)
        raw_output = model.outputs[0].asnumpy()
        pred = Softmax(raw_output)
        
        val_acc += CalAcc(pred, label.asnumpy()) /batch_size
        num_samp += 1
    if  nn>0:
        print('Alpha gradien being 0 :', nn)
    return(val_acc / num_samp)

'''
# # Generate Fixed Perturbed Data

# In[7]:

data = mx.symbol.Variable('data')
# first conv
conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=20)
tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max",
                              kernel=(2,2), stride=(2,2))
# second conv
conv2 = mx.symbol.Convolution(data=pool1, kernel=(5,5), num_filter=50)
tanh2 = mx.symbol.Activation(data=conv2, act_type="tanh")
pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max",
                              kernel=(2,2), stride=(2,2))
# first fullc
flatten = mx.symbol.Flatten(data=pool2)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=100)
tanh3 = mx.symbol.Activation(data=fc1, act_type="tanh")
# second fullc
fc2 = mx.symbol.FullyConnected(data=tanh3, num_hidden=10)


# In[8]:

data_shape = (batch_size, 1, 28, 28)
arg_names = fc2.list_arguments() # 'data' 
arg_shapes, output_shapes, aux_shapes = fc2.infer_shape(data=data_shape)

arg_arrays = [mx.nd.zeros(shape, ctx=dev) for shape in arg_shapes]
grad_arrays = [mx.nd.zeros(shape, ctx=dev) for shape in arg_shapes]
reqs = ["write" for name in arg_names]

model = fc2.bind(ctx=dev, args=arg_arrays, args_grad = grad_arrays, grad_req=reqs)
arg_map = dict(zip(arg_names, arg_arrays))
grad_map = dict(zip(arg_names, grad_arrays))
data_grad = grad_map["data"]
out_grad = mx.nd.zeros(model.outputs[0].shape, ctx=dev)


# In[9]:

for name in arg_names:
    if "weight" in name:
        arr = arg_map[name]
        arr[:] = mx.rnd.uniform(-0.07, 0.07, arr.shape)


# In[ ]:

num_round = 30
train_acc = 0.
nbatch = 0
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
    print("Train Accuracy: %.4f\t Train Loss: %.5f" % (train_acc, train_loss))


# In[ ]:

val_iter.reset()
val_acc = 0.0
val_acc_pb = 0.0
coe_pb = 1.5
num_samp = 0

perb_data = []
perb_lab = []

for dbatch in val_iter:
    data = dbatch.data[0]
    label = dbatch.label[0]
    arg_map["data"][:] = data    
    
    model.forward(is_train=True)
    theta = model.outputs[0].asnumpy()
    alpha = Softmax(theta)
    val_acc += CalAcc(alpha, label.asnumpy()) 
    #########
    grad = LogLossGrad(alpha, label.asnumpy())
    out_grad[:] = grad
    model.backward([out_grad])
    noise = data_grad.asnumpy()
    for j in range(batch_size):
        noise[j] = noise[j]/np.linalg.norm(noise[j].flatten(),2)
    pdata = data.asnumpy() + coe_pb * noise
    arg_map["data"][:] = pdata
    model.forward(is_train=True)
    raw_output = model.outputs[0].asnumpy()
    pred = Softmax(raw_output)
    val_acc_pb += CalAcc(pred, label.asnumpy()) 
    num_samp += batch_size
    
    perb_data.append(pdata)
    perb_lab.append(label.asnumpy())
print("Val Batch Accuracy: ", val_acc / num_samp)
print("Val Batch Accuracy after pertubation: ", val_acc_pb / num_samp)


# In[ ]:

pdata = np.concatenate(perb_data, axis = 0)
plabel = np.concatenate(perb_lab, axis = 0)
perb_iter = mx.io.NDArrayIter(
    data = pdata,
    label = plabel,
    batch_size = 100,
    shuffle = False    
)


# In[ ]:

perb_iter.reset()
num_samp = 0
val_acc = 0.0
for dbatch in perb_iter:
    data = dbatch.data[0]
    label = dbatch.label[0]
    arg_map["data"][:] = data    
    
    model.forward(is_train=True)
    theta = model.outputs[0].asnumpy()
    alpha = Softmax(theta)
    val_acc += CalAcc(alpha, label.asnumpy()) 
    num_samp += batch_size
print("Val Batch Accuracy after pertubation: ", val_acc / num_samp)


# # Normal Training

# In[ ]:

data = mx.symbol.Variable('data')
# first conv
conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=20)
tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max",
                              kernel=(2,2), stride=(2,2))
# second conv
conv2 = mx.symbol.Convolution(data=pool1, kernel=(5,5), num_filter=50)
tanh2 = mx.symbol.Activation(data=conv2, act_type="tanh")
pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max",
                              kernel=(2,2), stride=(2,2))
# first fullc
flatten = mx.symbol.Flatten(data=pool2)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=100)
tanh3 = mx.symbol.Activation(data=fc1, act_type="tanh")
# second fullc
fc2 = mx.symbol.FullyConnected(data=tanh3, num_hidden=10)


# In[ ]:

data_shape = (batch_size, 1, 28, 28)
arg_names = fc2.list_arguments() # 'data' 
arg_shapes, output_shapes, aux_shapes = fc2.infer_shape(data=data_shape)

arg_arrays = [mx.nd.zeros(shape, ctx=dev) for shape in arg_shapes]
grad_arrays = [mx.nd.zeros(shape, ctx=dev) for shape in arg_shapes]
reqs = ["write" for name in arg_names]

model = fc2.bind(ctx=dev, args=arg_arrays, args_grad = grad_arrays, grad_req=reqs)
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

num_round = 30
train_acc = 0.
nbatch = 0
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
    val_acc = acc_normal(model, val_iter,arg_map, grad_map)
    print("Train Accuracy: %.4f\t Val Accuracy: %.4f\t Train Loss: %.5f" % (train_acc, val_acc, train_loss))


# In[ ]:

num_round = 20
train_acc = 0.
nbatch = 0
lr = 0.01
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
    val_acc = acc_normal(model, val_iter,arg_map, grad_map)
    print("Train Accuracy: %.4f\t Val Accuracy: %.4f\t Train Loss: %.5f" % (train_acc, val_acc, train_loss))


# In[ ]:

for i in range(10,90,10):
  n=int(np.floor(i*784/100))
  print('KS perturbation: %.4f' % acc_perb_ks(model,n, val_iter, 1.5,arg_map, grad_map))

print('Normal Validation: %.3f' % acc_normal(model,val_iter,arg_map, grad_map))
#print('Fixed set perturbation: %.3f' % acc_normal(model, perb_iter,arg_map, grad_map))
print('L0 perturbation: %.3f' % acc_perb_L0(model, val_iter, 1.5,arg_map, grad_map))
print('L2 perturbation: %.3f' % acc_perb_L2(model, val_iter, 1.5,arg_map, grad_map))
print('Alpha perturbation: %.3f' % acc_perb_alpha(model, val_iter, 1.5,arg_map, grad_map))


# # Dropout Training

# In[ ]:

data = mx.symbol.Variable('data')
# first conv
conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=20)
tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max",
                              kernel=(2,2), stride=(2,2))
# second conv
conv2 = mx.symbol.Convolution(data=pool1, kernel=(5,5), num_filter=50)
tanh2 = mx.symbol.Activation(data=conv2, act_type="tanh")
pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max",
                              kernel=(2,2), stride=(2,2))
# first fullc
flatten = mx.symbol.Flatten(data=pool2)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=200)
tanh3 = mx.symbol.Activation(data=fc1, act_type="tanh")
dropout1 = mx.symbol.Dropout(data=tanh3, p=0.5)
# second fullc
fc2 = mx.symbol.FullyConnected(data=dropout1, num_hidden=10)


# In[ ]:

data_shape = (batch_size, 1, 28, 28)
arg_names = fc2.list_arguments() # 'data' 
arg_shapes, output_shapes, aux_shapes = fc2.infer_shape(data=data_shape)

arg_arrays = [mx.nd.zeros(shape, ctx=dev) for shape in arg_shapes]
grad_arrays = [mx.nd.zeros(shape, ctx=dev) for shape in arg_shapes]
reqs = ["write" for name in arg_names]

model = fc2.bind(ctx=dev, args=arg_arrays, args_grad = grad_arrays, grad_req=reqs)
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

num_round = 45
train_acc = 0.
nbatch = 0
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
    val_acc = acc_normal(model, val_iter,arg_map, grad_map)
    print("Train Accuracy: %.4f\t Val Accuracy: %.4f\t Train Loss: %.5f" % (train_acc, val_acc, train_loss))


# In[ ]:

num_round = 40
train_acc = 0.
nbatch = 0
lr = 0.01
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
    val_acc = acc_normal(model, val_iter,arg_map, grad_map)
    print("Train Accuracy: %.4f\t Val Accuracy: %.4f\t Train Loss: %.5f" % (train_acc, val_acc, train_loss))


# In[ ]:

print('Normal Validation: %.4f' % acc_normal(model,val_iter,arg_map, grad_map))
print('Fixed set perturbation: %.4f' % acc_normal(model, perb_iter,arg_map, grad_map))
print('L0 perturbation: %.4f' % acc_perb_L0(model, val_iter, 1.5,arg_map, grad_map))
print('L2 perturbation: %.4f' % acc_perb_L2(model, val_iter, 1.5,arg_map, grad_map))
print('Alpha perturbation: %.4f' % acc_perb_alpha(model, val_iter, 1.5,arg_map, grad_map))

'''
# # Ian's Method

# In[ ]:

data = mx.symbol.Variable('data')
# first conv
conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=20)
tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max",
                              kernel=(2,2), stride=(2,2))
# second conv
conv2 = mx.symbol.Convolution(data=pool1, kernel=(5,5), num_filter=50)
tanh2 = mx.symbol.Activation(data=conv2, act_type="tanh")
pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max",
                              kernel=(2,2), stride=(2,2))
# first fullc
flatten = mx.symbol.Flatten(data=pool2)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=200)
tanh3 = mx.symbol.Activation(data=fc1, act_type="tanh")
dropout1 = mx.symbol.Dropout(data=tanh3, p=0.5)
# second fullc
fc2 = mx.symbol.FullyConnected(data=dropout1, num_hidden=10)


# In[ ]:

data_shape = (batch_size, 1, 28, 28)
arg_names = fc2.list_arguments() # 'data' 
arg_shapes, output_shapes, aux_shapes = fc2.infer_shape(data=data_shape)

arg_arrays = [mx.nd.zeros(shape, ctx=dev) for shape in arg_shapes]
grad_arrays = [mx.nd.zeros(shape, ctx=dev) for shape in arg_shapes]
sum_arrays = [mx.nd.zeros(shape, ctx=dev) for shape in arg_shapes]

reqs = ["write" for name in arg_names]

model = fc2.bind(ctx=dev, args=arg_arrays, args_grad = grad_arrays, grad_req=reqs)
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

num_round = 45
train_acc = 0.
nbatch = 0
coe_pb = 1.75
lr= 0.05
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
    val_acc = acc_normal(model, val_iter,arg_map, grad_map)
    print("Train Accuracy: %.4f\t Val Accuracy: %.4f\t Train Loss: %.5f" % (train_acc, val_acc, train_loss))

'''
# In[ ]:

num_round = 40
train_acc = 0.
nbatch = 0
coe_pb = 1.75
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
    val_acc = acc_normal(model, val_iter,arg_map, grad_map)
    print("Train Accuracy: %.4f\t Val Accuracy: %.4f\t Train Loss: %.5f" % (train_acc, val_acc, train_loss))
'''

# In[ ]:

for i in range(10,90,10):
  n=int(np.floor(i*784/100))
  print('KS perturbation: %.4f' % acc_perb_ks(model,n, val_iter, 1.5,arg_map, grad_map))

print('Normal Validation: %.3f' % acc_normal(model,val_iter,arg_map, grad_map))
#print('Fixed set perturbation: %.3f' % acc_normal(model, perb_iter,arg_map, grad_map))
print('L0 perturbation: %.3f' % acc_perb_L0(model, val_iter, 1.5,arg_map, grad_map))
print('L2 perturbation: %.3f' % acc_perb_L2(model, val_iter, 1.5,arg_map, grad_map))
print('Alpha perturbation: %.3f' % acc_perb_alpha(model, val_iter, 1.5,arg_map, grad_map))

'''

# # LWA

# In[ ]:

data = mx.symbol.Variable('data')
# first conv
conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=20)
tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max",
                              kernel=(2,2), stride=(2,2))
# second conv
conv2 = mx.symbol.Convolution(data=pool1, kernel=(5,5), num_filter=50)
tanh2 = mx.symbol.Activation(data=conv2, act_type="tanh")
pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max",
                              kernel=(2,2), stride=(2,2))
# first fullc
flatten = mx.symbol.Flatten(data=pool2)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=400)
tanh3 = mx.symbol.Activation(data=fc1, act_type="tanh")
dropout1 = mx.symbol.Dropout(data=tanh3, p=0.5)

# second fullc
fc2 = mx.symbol.FullyConnected(data=dropout1, num_hidden=10)


# In[ ]:

data_shape = (batch_size, 1, 28, 28)
arg_names = fc2.list_arguments() # 'data' 
arg_shapes, output_shapes, aux_shapes = fc2.infer_shape(data=data_shape)

arg_arrays = [mx.nd.zeros(shape, ctx=dev) for shape in arg_shapes]
grad_arrays = [mx.nd.zeros(shape, ctx=dev) for shape in arg_shapes]
reqs = ["write" for name in arg_names]

model = fc2.bind(ctx=dev, args=arg_arrays, args_grad = grad_arrays, grad_req=reqs)
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

num_round = 60
train_acc = 0.
nbatch = 0
coe_pb = 1.2
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
        
        noise = data_grad.asnumpy()
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
    val_acc = acc_normal(model, val_iter,arg_map, grad_map)
    print("Train Accuracy: %.4f\t Val Accuracy: %.4f\t Train Loss: %.5f" % (train_acc, val_acc, train_loss))


# In[ ]:

num_round = 60
train_acc = 0.
nbatch = 0
coe_pb = 1.2
lr = 0.01
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
        
        noise = data_grad.asnumpy()
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
    val_acc = acc_normal(model, val_iter,arg_map, grad_map)
    print("Train Accuracy: %.4f\t Val Accuracy: %.4f\t Train Loss: %.5f" % (train_acc, val_acc, train_loss))


# In[ ]:

print('Normal Validation: %.4f' % acc_normal(model,val_iter,arg_map, grad_map))
print('Fixed set perturbation: %.4f' % acc_normal(model, perb_iter,arg_map, grad_map))
print('L0 perturbation: %.4f' % acc_perb_L0(model, val_iter, 1.5,arg_map, grad_map))
print('L2 perturbation: %.4f' % acc_perb_L2(model, val_iter, 1.5,arg_map, grad_map))
print('Alpha perturbation: %.4f' % acc_perb_alpha(model, val_iter, 1.5,arg_map, grad_map))


# In[ ]:

'''

