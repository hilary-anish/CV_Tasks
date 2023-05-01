# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 21:05:22 2023

@author: ANISH HILARY
"""

"""
Write a code using pytorch to replicate a grouped 2D convolution layer based on the original 2D convolution. 

The common way of using grouped 2D convolution layer in Pytorch is to use 
torch.nn.Conv2d(groups=n), where n is the number of groups.

However, it is possible to use a stack of n torch.nn.Conv2d(groups=1) to replicate the same
result. The wights must be copied and be split between the convs in the stack.

You can use:
    - use default values for anything unspecified  
    - all available functions in NumPy and Pytorch
    - the custom layer must be able to take all parameters of the original nn.Conv2d 
"""

import numpy as np
import torch
import torch.nn as nn


torch.manual_seed(8)    # DO NOT MODIFY!
np.random.seed(8)   # DO NOT MODIFY!

# random input (batch, channels, height, width)
x = torch.randn(2, 64, 100, 100)

# original 2d convolution
grouped_layer = nn.Conv2d(64, 128, 3, stride=1, padding=1, groups=16, bias=True)

# weights and bias
w_torch = grouped_layer.weight
b_torch = grouped_layer.bias

y = grouped_layer(x)

# now write your custom layer
'''
According to pytorch Documentation of Conv2D - group parameter

At groups=2, the operation becomes equivalent to having two conv layers side by side,
each seeing half the input channels and producing half the output channels,
and both subsequently concatenated.

So, For each group: new_input_channel = input_channel/num_group ;
 new_output_channel = output_channel/num_group
'''

class CustomGroupedConv2D(nn.Module):
    def __init__(self, num_groups, in_features, out_features, kernel_dim,
                 stride_count=1, padding_layer=1, dilate=1,  bias=True,
                 padding_mode='zeros', device=None, dtype=None):
        super(CustomGroupedConv2D, self).__init__()
        
        self.grouped_layer = nn.ModuleList()
        self.num_groups = num_groups
        for group in range(num_groups):
            self.grouped_layer.append(nn.Conv2d(in_channels=in_features//num_groups,
                                                out_channels = out_features//num_groups,
                                                kernel_size=kernel_dim, stride=stride_count,
                                                padding=padding_layer, dilation=dilate,bias=bias,
                                                padding_mode=padding_mode, device=device,dtype=dtype))
        
        self.weight = w_torch
        self.bias = b_torch
        
    def forward(self, x):
        b,c,h,w = x.size()
        x = x.reshape(b,self.num_groups,c//self.num_groups,h,w)

 # weight_copy       
        weight_chunk = torch.chunk(self.weight,self.num_groups,dim=0)
        for e, wgt_grp in enumerate(weight_chunk):
            self.grouped_layer[e].weight = nn.Parameter(wgt_grp.data)
 # bias_copy           
        bias_chunk = torch.chunk(self.bias,self.num_groups,dim=0)
        for e, bias_grp in enumerate(bias_chunk):
            self.grouped_layer[e].bias = nn.Parameter(bias_grp.data)
        
 # input passed through the model    
        layer_output = []
        for group_num in range(self.num_groups):
            layer_output.append(self.grouped_layer[group_num](x[:,group_num,:,:,:]))
            
 # concatenate all group outputs convolved with different Conv2d layers    
        output = torch.cat(layer_output,dim=1)
        
        return output
    
    
if __name__ == '__main__':

    custom_layer = CustomGroupedConv2D(num_groups=16,in_features=64,out_features=128,kernel_dim=3)
    y_custom_group = custom_layer(x)
    
     # Evaluating output of grouped_layer(x) and CustomGroupedConv2D(x)
     
    print(torch.isclose(y,y_custom_group))
    
    # the output of CustomGroupedConv2D(x) must be equal to grouped_layer(x)
