from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from binaryNet import Binary_W, Binary
import shutil
import numpy as np
import matplotlib
from matplotlib.colors import hsv_to_rgb


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


data_folder = '../../data'

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

def convert_batch(inp):

    inp = inp.numpy().transpose((0, 2, 3, 1))

    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])

    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)

    inp = matplotlib.colors.rgb_to_hsv(inp)
    inp = inp.transpose((0, 3, 1, 2))
    return torch.FloatTensor(inp)

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

        out = self.bn1(out)
        out = self.relu1(out)

        #POOLING
        out = self.pool1(out)

        #SECOND
        out, w = self.binary_w(out, self.conv2)
        out = F.conv2d(out, w, padding=1)

        out = self.bn2(out)
        out = self.relu2(out)

        #POOLING
        out = self.pool2(out)

        #THIRD
        out, w = self.binary_w(out, self.conv3)
        out = F.conv2d(out, w, padding=1)

        out = self.bn3(out)
        out = self.relu3(out)

        #FOURTH
        out, w = self.binary_w(out, self.conv4)
        out = F.conv2d(out, w, padding=1)

        out = self.bn4(out)
        out = self.relu4(out)

        out = self.pool4(out)

        #FIFTH
        out, w = self.binary_w(out, self.conv5)
        out = F.conv2d(out, w, padding=1)

        out = self.bn5(out)
        out = self.relu5(out)

        #SIXTH
        out, w = self.binary_w(out, self.conv6)
        out = F.conv2d(out, w, padding=1)

        out = self.bn6(out)
        out = self.relu6(out)

        out = self.pool6(out)

        #SEVEN
        out, w = self.binary_w(out, self.conv7)
        out = F.conv2d(out, w, padding=1)

        out = self.bn7(out)
        out = self.relu7(out)

        #EIGHT
        out, w = self.binary_w(out, self.conv8)
        out = F.conv2d(out, w, padding=1)

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

def train(epoch):
    model.train()
    step = (epoch-1)*len(train_loader.dataset)/100
    for batch_idx, (data, target) in enumerate(train_loader):
        data = convert_batch(data)
        data = data[:, 0 : 1, :] * 360.0
        data = threshold(data,  from_limit, to_limit);
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
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

def adjust_learning_rate(lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
    lr = lr * (0.1 ** (epoch // 10))

    print ('Learning rate: ' + str(lr))
    # log to TensorBoard
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def test(epoch):
    global best_prec1
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data = convert_batch(data)
        data = data[:, 0: 1, :] * 360.0
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

thresholdsArray = [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360]
for i in range(0, len(thresholdsArray) - 1):
    print('Threshold No {}\n'.format(i+1))
    lr = 0.01

    model = VGG()
    if args.cuda:
        model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum)
    criterion = nn.CrossEntropyLoss()

    from_limit = thresholdsArray[i]
    to_limit = thresholdsArray[i+1]
    file_name = 'BINARY_'+ str(i+1) + '_' + str(len(thresholdsArray))
    best_prec1 = 0.0

    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(args.lr, optimizer, epoch)
        train(epoch)
        test(epoch)
