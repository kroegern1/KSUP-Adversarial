
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


# In[4]:

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

def acc_perb_urn(model, val_iter, coe_pb, arg_map, grad_map):
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
        grad = grad_map["data"].asnumpy()
        noise=np.random.randn(grad.shape[0],grad.shape[1],grad.shape[2],grad.shape[3])
        pdata = data.asnumpy() + coe_pb * noise
        arg_map["data"][:] = pdata
        model.forward(is_train=False)
        raw_output = model.outputs[0].asnumpy()
        pred = Softmax(raw_output)
        
        val_acc += CalAcc(pred, label.asnumpy()) /  batch_size 
        num_batch += 1
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


# # Ksup

# In[ ]:

# input
data = mx.symbol.Variable('data')
# first fullc
flatten = mx.symbol.Flatten(data=data)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=400)
relu1 = mx.symbol.Activation(data=fc1, act_type="relu")
fc2 = mx.symbol.FullyConnected(data=relu1, num_hidden=400)
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

num_round = 40
train_acc = 0.
nbatch = 0
coe_pb = 1.7
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
            
        n=int(np.floor(.25*784))
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
print('urn perturbation: %.3f' % acc_perb_urn(model, val_iter, coe_pb, arg_map, grad_map))


