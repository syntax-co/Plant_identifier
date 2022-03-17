import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as funct



class linear_network(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.device='cpu'
        
        
        self.conv_one=torch.nn.Conv2d(3,12,3).to(self.device)
        self.conv_two=torch.nn.Conv2d(12,24,3).to(self.device)
        self.conv_three=torch.nn.Conv2d(24,12,3).to(self.device)
        self.conv_four=torch.nn.Conv2d(12,10,3).to(self.device)
        
        self.test_input=torch.zeros(1,3,4000,6000).to(self.device)
        self.line_size=self.convs(self.test_input)
        
        self.line_one=nn.Linear(self.line_size,512).to(self.device)
        self.line_two=nn.Linear(512,14).to(self.device)
        
        if self.device=='cuda':
            self.test_input.to('cpu')
            del self.test_input
            torch.cuda.empty_cache()
        
        
        self.optimizer=optim.Adam(self.parameters())
        self.criterion=nn.MSELoss()#.to(self.device)
        
        
    
    def convs(self,x):
        x=funct.max_pool2d(funct.relu(self.conv_one(x)),(2,2))
        x=funct.max_pool2d(funct.relu(self.conv_two(x)),(2,2))
        x=funct.max_pool2d(funct.relu(self.conv_three(x)),(2,2))
        x=funct.max_pool2d(funct.relu(self.conv_four(x)),(2,2))
        
        
        size=x.shape
        self.zero_grad()
        return size[1]*size[2]*size[3]
        


    def forward(self,x):
        x=funct.max_pool2d(funct.relu(self.conv_one(x)),(2,2))
        x=funct.max_pool2d(funct.relu(self.conv_two(x)),(2,2))
        x=funct.max_pool2d(funct.relu(self.conv_three(x)),(2,2))
        x=funct.max_pool2d(funct.relu(self.conv_four(x)),(2,2))
        
        x=torch.flatten(x)
        
        x=funct.relu(self.line_one(x))
        x=self.line_two(x)
        x=funct.softmax(x,dim=0)
        return x
        
    def toggle_device(self):
        
        if self.device=='cpu':
            self.device='cuda'
            
            self.conv_one.to('cuda')
            self.conv_two.to('cuda')
            self.conv_three.to('cuda')
            self.conv_four.to('cuda')
            self.line_one.to('cuda')
            self.line_two.to('cuda')
            
            
            
            
        
        elif self.device=='cuda':
            self.device='cpu'
        
            self.conv_one.to('cpu')
            self.conv_two.to('cpu')
            self.conv_three.to('cpu')
            self.conv_four.to('cpu')
            self.line_one.to('cpu')
            self.line_two.to('cpu')
        
            torch.cuda.empty_cache()
        
        
        
        # self.criterion.to(self.device)
        
        
        
        
    def save(self):
        torch.save(self.state_dict,'leaf_model.pth')
        
        
    
        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        