# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 21:19:21 2023

@author: ANISH HILARY
"""

"""
develop a model based on the onnx file in model/model.onnx 

Note:
    - initialize the convolutions layer with uniform xavier
    - initialize the linear layer with a normal distribution (mean=0.0, std=1.0)
    - initialize all biases with zeros
    - use batch norm wherever is relevant
    - use random seed 8
    - use default values for anything unspecified
"""

import numpy as np
import torch
import torch.nn as nn
import onnx


torch.manual_seed(8)    # DO NOT MODIFY!
np.random.seed(8)   # DO NOT MODIFY!


# write your code here ...

'''loading and printing the onnx file'''

model_dir = './model/model.onnx'
onnx_model = onnx.load(model_dir)
onnx.checker.check_model(onnx_model)
print(f'onnx graph: {onnx.helper.printable_graph(onnx_model.graph)}')




class act(nn.Module):
    def __init__(self):
        super(act, self).__init__()
        self.Sigmoid_output_0 = nn.Sigmoid()
        
    def forward(self,x):
        
        return self.Sigmoid_output_0(x)
  
    
class linear(nn.Module):
    def __init__(self):
        super(linear, self).__init__()
        
        self.linear = nn.Linear(256,256, bias = True)
        
    def forward(self,x):
        
        x = self.linear(x)
        
        return x
        



class trans1(nn.Module):
    def __init__(self):
        super(trans1, self).__init__()
        
        self.mp = nn.MaxPool2d(kernel_size=2,stride=2,padding=0,ceil_mode=False)
        self.conv1 = nn.Conv2d(256, 128, 1,stride=1, padding=0,dilation=1,groups=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.sigmoid = act()
        self.conv2 = nn.Conv2d(256, 128, 1,stride=1, padding=0,dilation=1,groups=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, 3,stride=2, padding=1,dilation=1,groups=1)
        self.bn3 = nn.BatchNorm2d(128)
    
    def forward(self,x):
  # maxpool (uses input)
        x_maxpool = self.mp(x)
        
        x_conv1 = self.conv1(x_maxpool)
        x_conv1 = self.bn1(x_conv1)
        x_sig = self.sigmoid(x_conv1)
        x_conv1 = torch.mul(x_conv1, x_sig) 
  # conv2 (uses input)      
        x_conv2 = self.conv2(x)
        x_conv2 = self.bn1(x_conv2)
        x_sig = self.sigmoid(x_conv2)
        x_conv2 = torch.mul(x_conv2, x_sig)
        
        x_conv3 = self.conv3(x_conv2)
        x_conv3 = self.bn1(x_conv3)
        x_sig = self.sigmoid(x_conv3)
        x_conv3 = torch.mul(x_conv3, x_sig)
        
# concatenation
        x_concat = torch.cat((x_conv3,x_conv1),dim=1)
        
        return x_concat



class elan1(nn.Module):
    def __init__(self):
        super(elan1, self).__init__()
        
        self.conv1 = nn.Conv2d(128, 64, 1,stride=1, padding=0,dilation=1,groups=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.sigmoid = act()
        self.conv2 = nn.Conv2d(128, 64, 1,stride=1, padding=0,dilation=1,groups=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3,stride=1, padding=1,dilation=1,groups=1) 
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3,stride=1, padding=1,dilation=1,groups=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, 3,stride=1, padding=1,dilation=1,groups=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, 3,stride=1, padding=1,dilation=1,groups=1)
        self.bn6 = nn.BatchNorm2d(64)
        
        self.conv7 = nn.Conv2d(256, 256, 1,stride=1, padding=0,dilation=1,groups=1)
        self.bn7 = nn.BatchNorm2d(256)
        
    def forward(self,x):
        x_conv1 = self.conv1(x)
        x_conv1 = self.bn1(x_conv1)
        x_sig = self.sigmoid(x_conv1)
        x_conv1 = torch.mul(x_conv1, x_sig)
        
        x_conv2 = self.conv2(x)
        x_conv2 = self.bn2(x_conv2)
        x_sig = self.sigmoid(x_conv2)
        x_conv2 = torch.mul(x_conv2, x_sig)
        
        x_conv3 = self.conv3(x_conv2)
        x_conv3 = self.bn3(x_conv3)
        x_sig = self.sigmoid(x_conv3)
        x_conv3 = torch.mul(x_conv3, x_sig)
        
        x_conv4 = self.conv4(x_conv3)
        x_conv4 = self.bn4(x_conv4)
        x_sig = self.sigmoid(x_conv4)
        x_conv4 = torch.mul(x_conv4, x_sig)
        
        x_conv5 = self.conv5(x_conv4)
        x_conv5 = self.bn5(x_conv5)
        x_sig = self.sigmoid(x_conv5)
        x_conv5 = torch.mul(x_conv5, x_sig)
        
        x_conv6 = self.conv6(x_conv5)
        x_conv6 = self.bn6(x_conv6)
        x_sig = self.sigmoid(x_conv6)
        x_conv6 = torch.mul(x_conv6, x_sig)
  # concatenation 
        x_concat = torch.cat((x_conv6,x_conv4,x_conv2,x_conv1),dim=1)    
        
        x_conv7 = self.conv7(x_concat)
        x_conv7 = self.bn7(x_conv7)
        x_sig = self.sigmoid(x_conv7)
        x_conv7 = torch.mul(x_conv7, x_sig)
        
        return x_conv7
        
        
        

class Custom_Model(nn.Module):
    def __init__(self):
        super(Custom_Model, self).__init__()
        
        self.conv0 = nn.Conv2d(3, 32, 3,stride=1, padding=1,dilation=1,groups=1)
        self.bn0 = nn.BatchNorm2d(32)
        self.sigmoid = act()
        self.conv1 = nn.Conv2d(32, 64, 3,stride=2, padding=1,dilation=1,groups=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3,stride=1, padding=1,dilation=1,groups=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3,stride=2, padding=1,dilation=1,groups=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.elan1 = elan1()
        self.trans1 = trans1()
        self.linear = linear()
        
    def forward(self,x):
        x = self.conv0(x)
        x = self.bn0(x)
        x_sig = self.sigmoid(x)
        x = torch.mul(x, x_sig)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x_sig = self.sigmoid(x)
        x = torch.mul(x, x_sig)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x_sig = self.sigmoid(x)
        x = torch.mul(x, x_sig)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x_sig = self.sigmoid(x)
        x = torch.mul(x, x_sig)
        
        elan1_x = self.elan1(x)
        
        tran1_x = self.trans1(elan1_x)
        
# permute
        x_transpose_0 = tran1_x.permute(0,2,3,1)
        b,h,w,c = x_transpose_0.shape
# constant_0       
        x_constant_0 = torch.randn(256)
# reshape_0 
        x_out_0 = x_transpose_0*x_constant_0
        x_reshape_0 = x_out_0.view(-1,256)
        
# linear        
        lin_x = self.linear(x_reshape_0) 
# constant_1      
        x_constant_1 = torch.randn(256)
# reshape_1
        x_out_1 = lin_x*x_constant_1
# convert the linear layer back to 2-D image
        x_reshape_1 = x_out_1.view(b,h,w,c)
# permute to get channel in dim=1
        x_transpose_1 = x_reshape_1.permute(0,3,1,2)
        
        output = nn.Sigmoid()(x_transpose_1)
        
        return output
    
        
        
if __name__ == '__main__':
    mod = Custom_Model()
    
    print(mod)
# no need to use weight.data, 
# as per the documentation the function runs in torch.no_grad() mode

    for module in mod.modules():
        if isinstance(module,nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        if isinstance(module,nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=1.0)
            nn.init.zeros_(module.bias)
            
    input_ = torch.randn(1,3,160,320)
    model_output = mod(input_)
    
    print(f'Model_Input_shape: {input_.shape}, Model_Output_shape: {model_output.shape}')
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    