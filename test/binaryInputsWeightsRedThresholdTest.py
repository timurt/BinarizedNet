from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import colormath
from torchvision import datasets, transforms
from torch.autograd import Variable
from logger import Logger
from binaryNet import Binary_W, Binary, Threshold
import shutil



# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=40, metavar='N',
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

#Out of 15
#-0.4242129623889923
#-0.207832984129588
#0.008546994129816
#0.22492697238922
#0.441306950648624
#0.657686928908028
#0.874066907167432
#1.090446885426836
#1.30682686368624
#1.523206841945644
#1.739586820205048
#1.955966798464452
#2.172346776723856
#2.38872675498326
#2.605106733242664
#2.821486711502075




data_folder = './data'

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(data_folder, train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(data_folder, train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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

def threshold(data, lowerBound, upperBound):
    INF = -100
    output = data.clone()
    output[lowerBound > output] = INF
    output[upperBound <= output] = INF
    output[output != INF] = 1
    output[output == INF] = -1
    return output

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename + '_latest.pth.tar')
    if is_best:
        shutil.copyfile(filename + '_latest.pth.tar', filename + '_best.pth.tar')

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU(inplace=True)

        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU(inplace=True)

        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.relu6 = nn.ReLU(inplace=True)

        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.relu7 = nn.ReLU(inplace=True)

        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.relu8 = nn.ReLU(inplace=True)

        self.pool8 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.avg = nn.AvgPool2d(kernel_size=1, stride=1);
        self.fc = nn.Linear(512, 512)
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        #FIRST
        out, w = self.binary_w(x, self.conv1)
        out = F.conv2d(out, w, padding=1)

        #out = self.conv1(x)
        #print(out.data.numpy().shape)
        #print('After')
        out = self.bn1(out)
        out = self.relu1(out)

        #POOLING
        out = self.pool1(out)

        #SECOND
        out, w = self.binary_w(out, self.conv2)
        out = F.conv2d(out, w, padding=1)

        #out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        #POOLING
        out = self.pool2(out)

        #THIRD
        out, w = self.binary_w(out, self.conv3)
        out = F.conv2d(out, w, padding=1)

        #out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)

        #FOURTH
        out, w = self.binary_w(out, self.conv4)
        out = F.conv2d(out, w, padding=1)

        # out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu4(out)

        out = self.pool4(out)

        #FIFTH
        out, w = self.binary_w(out, self.conv5)
        out = F.conv2d(out, w, padding=1)

        #out = self.conv5(out)
        out = self.bn5(out)
        out = self.relu5(out)

        #SIXTH
        out, w = self.binary_w(out, self.conv6)
        out = F.conv2d(out, w, padding=1)

        # out = self.conv6(out)
        out = self.bn6(out)
        out = self.relu6(out)

        out = self.pool6(out)

        #SEVEN
        out, w = self.binary_w(out, self.conv7)
        out = F.conv2d(out, w, padding=1)

        #out = self.conv7(out)
        out = self.bn7(out)
        out = self.relu7(out)

        #EIGHT
        out, w = self.binary_w(out, self.conv8)
        out = F.conv2d(out, w, padding=1)

        # out = self.conv8(out)
        out = self.bn8(out)
        out = self.relu8(out)

        out = self.pool8(out)

        out = self.avg(out)

        out = self.binary(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.classifier(out)
        return out

    def binary(self, input):
        return Binary()(input)

    def binary_w(self, input, param):
        return Binary_W()(input, param.weight)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.bn0 = nn.BatchNorm2d(3)
        self.bn1 = nn.BatchNorm2d(10)
        self.bn2 = nn.BatchNorm2d(20)
        self.bn3 = nn.BatchNorm1d(50)

    def forward(self, x):
        im = x
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x, im

    def forward_old(self, x):
        #x = self.th(x, self.t)
        im = x
        x,w = self.binary_w(x, self.conv1)
        x = F.conv2d(x,w)
        x = F.tanh(F.max_pool2d(self.bn1(x), 2))
        x,w = self.binary_w(x,self.conv2)
        x = F.conv2d(x,w)
        x = F.tanh(F.max_pool2d(self.bn2(x), 2))
        x = self.binary(x)
        #print(x.data.numpy().shape)
        x = x.view(-1, 25*20)
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


# Set the logger
#logger = Logger('./logs')


#optimizer = optim.Adam(model.parameters(), lr=args.lr)

i = 0
def test(epoch):
    global best_prec1
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:

        data = data[:, 0: 1, :]
        data = threshold(data, from_limit, to_limit);
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += criterion(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader) # loss function already averages over batch size

    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), Best ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy, best_prec1))

    is_best = accuracy > best_prec1
    best_prec1 = max(accuracy, best_prec1)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
    }, is_best, file_name)

thresholdsArray = [-0.4242129623889923, -0.063579665289985, 0.297053631809022,
                   0.657686928908029, 1.018320226007036, 1.378953523106043,
                   1.73958682020505, 2.100220117304057, 2.460853414403064, 2.821486711502075]
for i in range(0, len(thresholdsArray) - 1):
    print('Threshold No {}\n'.format(i+1))
    lr = 0.01

    file_name = 'RED_BINARY_' + str(i + 1) + '_9_best.pth.tar'

    model = VGG()

    if args.cuda:
        model.cuda()

    checkpoint = torch.load(file_name)
    best_prec1 = checkpoint['best_prec1']
    model.load_state_dict(checkpoint['state_dict'])

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum)
    criterion = nn.CrossEntropyLoss()

    from_limit = thresholdsArray[i]
    to_limit = thresholdsArray[i + 1]

    print('Best precision for threshold {}, is {}\n'.format((i+1), best_prec1))
    test(1)

    
#torch.save(model, 'binary_mnist_l.pth.tar')
