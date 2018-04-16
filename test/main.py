from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from logger import Logger
from binaryNet import Binary_W, Binary, Threshold

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=12, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

data_folder = '../data'
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(data_folder, train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(data_folder, train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
    
    
    
#kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}
#train_loader = torch.utils.data.DataLoader(
#    datasets.CIFAR10(data_folder, train=True, download=True,
#                   transform=transforms.Compose([
#                       transforms.RandomCrop([28, 28]),
#                       transforms.ToTensor(),
#                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#                       
#                   ])),
#    batch_size=args.batch_size, shuffle=True, **kwargs)
#test_loader = torch.utils.data.DataLoader(
#    datasets.CIFAR10(data_folder, train=False, transform=transforms.Compose([
#                       transforms.RandomCrop([28, 28]),
#                       transforms.ToTensor(),
#                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#                       
#                   ])),
#    batch_size=args.batch_size, shuffle=True, **kwargs)

def to_np(x):
    return x.data.cpu().numpy()

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)    

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.t = 2
        self.conv1 = nn.Conv2d(self.t, 10, kernel_size=5)
       # self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(16*20, 50)
        self.fc2 = nn.Linear(50, 10)
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(10)
        self.bn2 = nn.BatchNorm2d(20)
        self.bn3 = nn.BatchNorm1d(50)

    def forward(self, x):
        
        x = self.th(x, self.t)
        im = x
        x,w = self.binary_w(x, self.conv1)
        x = F.conv2d(x,w)
        x = F.tanh(F.max_pool2d(self.bn1(x), 2))
        x,w = self.binary_w(x,self.conv2)
        x = F.conv2d(x,w)
        x = F.tanh(F.max_pool2d(self.bn2(x), 2))
        x = self.binary(x)
        x = x.view(-1, 16*20)
        x = F.tanh( self.bn3(self.fc1(x)))
    #    x = self.binary(x)
   
        x = self.fc2(x)
        
        return x, im
    
    def binary(self, input):
        return Binary()(input)  
    
    def binary_w(self, input, param):
       return Binary_W()(input, param.weight)
   
    def th(self, input, t):
        return Threshold(t)(input) 


model = Net()
if args.cuda:
    model.cuda()
    

if args.cuda:
    model.cuda()
    
# Set the logger
#logger = Logger('./logs')

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
#optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()
i = 0

def train(epoch):
    model.train()
    step = (epoch-1)*len(train_loader.dataset)/100
    for batch_idx, (data, target) in enumerate(train_loader):
        print(data.shape)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        print(data.shape)
        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()
        output, im1 = model(data)
        #loss = F.nll_loss(output, target)
        loss = criterion(output, target)
        if loss.data[0]<10.0:
            #print ('True')
            loss.backward()
            optimizer.step()
            
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.00f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
            
            # Compute accuracy
            _, argmax = torch.max(output, 1)
            accuracy = (target == argmax.squeeze()).float().mean()
#            #============ TensorBoard logging ============#
            # (1) Log the scalar values
#            info = {
#                'loss': loss.data[0],
#                'accuracy': accuracy.data[0]
#            }
#        
#            for tag, value in info.items():
#                logger.scalar_summary(tag, value, step+1)
##        
#            # (2) Log values and gradients of the parameters (histogram)
#            for tag, value in model.named_parameters():
#                tag = tag.replace('.', '/')
#                logger.histo_summary(tag, to_np(value), step+1)
#              #  logger.histo_summary(tag+'/grad', to_np(value.grad), step+1)
##        
#            # (3) Log the images
#            info = {
#                'images': to_np(im1.view(100,model.t, 28,28))[:10, 5:8, :, :]
#            }
#        
#            for tag, images in info.items():
#                logger.image_summary(tag, images, step+1)
               
                

                
                
def adjust_learning_rate(lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
    lr = lr * (0.1 ** (epoch // 13)) 

    print ('Learning rate: ' + str(lr))
    # log to TensorBoard
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr  
                   
        

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output, im1 = model(data)
        test_loss += criterion(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, args.epochs + 1):
  #  adjust_learning_rate(args.lr, optimizer, epoch)
    train(epoch)
    test(epoch)
    
#torch.save(model, 'binary_mnist_l.pth.tar')
