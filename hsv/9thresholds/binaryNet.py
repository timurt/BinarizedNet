import torch
from torch.autograd import Function

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
