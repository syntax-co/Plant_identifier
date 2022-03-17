
from Dataformer import dataformer
from model import linear_network
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as funct

import time,os,random,sys

class data_handler:
    def __init__ (self):
        self.former=dataformer()
        self.network=linear_network()
        self.leaf_key=['Alstonia Scholaris','Arjun',
                        'Bael','Basil','Chinar','Gauva',
                        'Jamun','Jatropha','Lemon','Mango',
                        'Pomegranate','Pongamia Pinnata']
        
        self.testing()
        
    def convert_output(self,netout):
        leafs=netout[0:12]
        status=netout[12:14]
        
    def convert_target(self,target):
        name=target[0]
        status=target[1]
        
        final=[0 for i in range(14)]
        
        dex=self.leaf_key.index(name)
        final[dex]=1
        
        if status=='healthy':
            final[12]=1
            
        elif status=='diseased':
            final[13]=1
        
        
        final=torch.FloatTensor(final)
        
        return final
        
        
        
    
    def get_images(self,image_info):
        
        images=[]
        for i in image_info:
            ipath=i[2]
            file=Image.open(ipath)
            image=np.asarray(file)
            image=np.moveaxis(image,-1,0)
            image=np.expand_dims(image,0)
            
            file.close()
            
            image=torch.FloatTensor(image)
            
            
            images.append(image)
            
        
        
        
    
        return images
        
        
    def train_images(self,image_info):
        images=image_info[0]
        labels=image_info[1]

        # print('------- batch progress')
        bsize=len(labels)
        counter=0
        for index in range(bsize):
            
            image=images[index]
            label=labels[index]
            height=image.shape[1]
            width=image.shape[2]
            batch=1
            
            
            self.network.to('cuda')
            
            holder=image.to('cuda')
            
            target=self.convert_target(label[0:2])
            
            
            
            result=self.network.forward(holder)
            result=result.detach().requires_grad_().to('cpu')
            
            self.network.zero_grad()
            loss=self.network.criterion(result,target)
            
            holder.to('cpu')
            del holder
            torch.cuda.empty_cache()
            
            loss.backward()
            self.network.optimizer.step()
            
            
            
    
    
    def testing(self):
        
        training_size=.75
        testing_size=1-training_size
        
        
        info=self.former.leaf_data()
        info=info['diseased']+info['healthy']
        isize=len(info)
        
        info=info[0:round(isize/2)]
        random.shuffle(info)
        
        
        testing_data=info[round(isize*training_size):isize]
        info=info[0:round(isize*training_size)]
        
        
        epochs=10
        batch_size=20
        full_batches=isize//batch_size
        over_batch=isize%batch_size
        
        
        for epoch in range(epochs):
            for i in range(full_batches):
                start=i*batch_size
                end=(i+1)*batch_size
                paths=info[start:end]
                images=self.get_images(paths)
                full_info=[images,paths]
                
                self.train_images(full_info)
                print('&'*50)
                print('traingin epoch {}'.format(epoch+1))
                print('trainging batch {}'.format(i+1))
                print('epoch progress',round(epoch/epochs,4)*100)
                print('batch progress',round(i/full_batches,4)*100)
                print('&'*50)
        
            self.network.save()
                
            
        
            
            
            
            
        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
if __name__=='__main__':
    handler=data_handler()
    
    