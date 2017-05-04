# Normal data

db1=data.asnumpy()
db=db1[1:11]
lb1=label.asnumpy()
lb=lb1[1:11]

for i in range(10):
   plt.subplot(1,10,i+1)
   plt.imshow(db[i].reshape((28,28)),cmap='Greys_r')
   plt.axis('off')


plt.show()


#out= np.argmax(pred, axis=1)

#L2
noise = grad_map["data"].asnumpy()

#L1
noise = np.sign(grad_map["data"].asnumpy())

###
for j in range(batch_size):
            if np.linalg.norm(noise[j].flatten(),2) ==0:
                nn+=1
            y = label.asnumpy()[j]
            if (y == np.argmax(alpha[j])):
                noise[j] = noise[j]/np.linalg.norm(noise[j].flatten(),2)
            else:
                noise[j] = 0




pdata = data.asnumpy() + coe_pb * noise


#KSUP

new_grad= grad_map["data"].asnumpy()
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
            if np.linalg.norm(noise[j].flatten(),2) ==0:
                nn+=1
            y = label.asnumpy()[j]
            if (y == np.argmax(alpha[j])): 
                noise[j] = noise[j]/np.linalg.norm(noise[j].flatten(),2)
            else:
                noise[j] = 0
        


pdata = data.asnumpy() + coe_pb * noise

#URN

grad = grad_map["data"].asnumpy()
noise=np.random.randn(grad.shape[0],grad.shape[1],grad.shape[2],grad.shape[3])
pdata = data.asnumpy() + coe_pb * noise


#Noised data


pd=pdata[1:11]
for i in range(10):
   plt.subplot(1,10,i+1)
   plt.imshow(pd[i].reshape((28,28)),cmap='Greys_r')
   plt.axis('off')


plt.show()


