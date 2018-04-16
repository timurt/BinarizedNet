import torch
from torch.autograd import Function
import numpy as np
#import torch.nn.functional as F

class Binary(Function):
    def __init__(self):
        super(Binary,self).__init__()

    
    def forward(self, input):
        output = torch.sign(input)
        
        self.save_for_backward(input)  
        
        return output


    def backward(self, grad_output):
#        input = self.saved_tensors
        
#        grad_input = torch.zeros(grad_output.size()).type_as(grad_output)
#        print input
#        print torch.ge(input,torch.Tensor(1))
#        grad_input[torch.ge(input,torch.Tensor(1))]=0
#        grad_input[torch.le(input,torch.Tensor(-1))]=0

        grad_input = grad_output
        
        return grad_input
    
    
class Binary_W(Function):
    def __init__(self):
        super(Binary_W,self).__init__()


    def forward(self, input, weight):
        
        new_weight = torch.sign(weight)
        new_input = torch.sign(input)
        self.save_for_backward(input, weight)
       # output = F.conv2d(new_input,new_weight)
        return  new_input, new_weight


    def backward(self, grad_input, grad_weight):
      #  input, weight = self.saved_tensors
       # print grad_input
        return grad_input, grad_weight
    
    
class Threshold(Function):
    def __init__(self, th):
        super(Threshold,self).__init__()
        self.th = th
    
    def forward(self, input):
        self.save_for_backward(input) 
        m = torch.min(input)
        n = torch.max(input)
        r = np.linspace(m, n, num=self.th+1, endpoint=False)
        for i in r[1:]: # excluding the smallest value 
                output = input.clone()
                output[output< i]=-1
                output[output>=i]= 1
               
                if i==r[1]:
                    out = output #torch.unsqueeze(output,0)
                else:
                    out = torch.cat([out,output],1) #torch.cat([out,torch.unsqueeze(output,0)],0) 
        return out


    def backward(self, grad_output):
#        input = self.saved_tensors
        
#        grad_input = torch.zeros(grad_output.size()).type_as(grad_output)
#        print input
#        print torch.ge(input,torch.Tensor(1))
#        grad_input[torch.ge(input,torch.Tensor(1))]=0
#        grad_input[torch.le(input,torch.Tensor(-1))]=0

        grad_input = grad_output
        
        return grad_input
