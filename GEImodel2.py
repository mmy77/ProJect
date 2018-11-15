
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tensorboardX import SummaryWriter
from time  import time
#from datagen inport MyDataset

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
plt.ion()   # 交互模式
use_cuda = torch.cuda.is_available()

root = '/home/xiaoqian/GeiGait/'
# -----------------ready the dataset--------------------------
def default_loader(path):
    return Image.open(path).convert('RGB')
class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],words[1],words[2]))  
        self.imgs = imgs
        self.transform = transforms.Compose([
            transforms.CenterCrop((300,300)),
            transforms.Grayscale(num_output_channels=1),
            #transforms.ToTensor()
            ])
        
        self.target_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            #transforms.ToTensor(),
            #transforms.Normalize(mean=(0.5,0.5),std=(0.5,0.5)),
            #transforms.ToPILImage()
            ])
        self.loader = loader

    def __getitem__(self, index):

        global global_point 
        global_point = global_point+1
        index = int(index%(self.__len__()))

        order, input_data, target_data = self.imgs[index]
        #print(fn)
        train_img = self.loader(input_data) 
        target_img = self.loader(target_data)  

       # train_img = np.expand_dims(np.array(train_img),axis=0)
        #target_img = np.expand_dims(np.array(target_img),axis=0)
        #print(train_img.shape)
        train_img = self.transform(train_img)
        target_img = self.target_transform(target_img)
        #print(input_data,target_data)
        train_img = np.array(train_img,np.float32)[np.newaxis,:]
        target_img = np.array(target_img,np.float32)[np.newaxis,:]

        train_img = train_img/255
        target_img = target_img/255
        #return np.array(train_img)[np.newaxis,:].astype(float), np.array(target_img)[np.newaxis,:].astype(float)
        return int(order),train_img,target_img
        #return fn, label

    def __len__(self):
        return len(self.imgs)
trainroot = '/mnt/disk50/datasets/dataset-gait/CASIA/DatasetB/silhouettes/'

train_data=MyDataset(txt=root+'datalist/train_ksgei.txt')
#print(img1[0].shape,torch.tensor(img1[0]))
test_data=MyDataset(txt=root+'datalist/test_ksgei.txt')

train_loader = torch.utils.data.DataLoader(train_data,batch_size=30, shuffle=True, num_workers=8,drop_last=True)
test_loader = torch.utils.data.DataLoader(test_data,batch_size=30,shuffle=False,num_workers=8,drop_last=True)
'''
# 训练集
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='.', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])), batch_size=64, shuffle=True, num_workers=4)
# 测试集
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='.', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])), batch_size=64, shuffle=True, num_workers=4)
'''
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Net(nn.Module):
    def __init__(self, block, layers, num_classes=100):
        self.inplanes = 64
        super(Net, self).__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=4,stride=2),#149,149
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(4, 8, kernel_size=4,stride=2),#36,36
            nn.MaxPool2d(2, stride=2),#18,18
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=4,stride=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        # 3 * 2 仿射矩阵 (affine matrix) 的回归器
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 4 * 4, 32),#10,4,4
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # 用身份转换 (identity transformation) 初始化权重 (weights) / 偏置 (bias)
        self.fc_loc[2].weight.data.fill_(0)
        self.fc_loc[2].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])
        #super
        self.conv1 = nn.Conv2d(1, 64, kernel_size=17, stride=1, padding=[0,20],bias=False)#128*128
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)#56*56
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)#56*56
        #self.maxpool = nn.MaxPool2d(kernel_size=[27,6], stride=1)
        #output 56x56
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)#28*28
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)#14*14
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)#7*7
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc1 = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    # 空间转换网络的前向函数 (Spatial transformer network forward function)
    def stn(self, x):
        xs = self.localization(x)
        #print(xs.size())

        xs = xs.view(-1, 10 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, torch.Size([x.size(0),1,128,88]))
        x = F.grid_sample(x, grid)

        return x
    '''
    def Gei(self,x):
        x = torch.transpose(x,0,1)

        fusion = nn.Conv2d(x.size(1),1,kernel_size=1,stride=1).cuda()
        fusion.weight.data.fill_(1/x.size(1))
        for i in fusion.parameters():
            i.requires_grad = False
        return fusion(x)
    '''
    def Gei(self,x):
        x = torch.transpose(x,0,1)
        weight = np.ones([int(x.size(1)),int(x.size(1)/10),1,1])
        weight = torch.from_numpy(weight).cuda().float()
        fusion = F.conv2d(x, weight,stride=1,groups=10).cuda()
        print(fusion.size())
        return torch.transpose(fusion,0,1)

    def forward(self, x):
        x = self.stn(x)
        #print(x.size()) #30,1,128,88
        return x
        x = self.Gei(x)
       
        x = self.conv1(x)
       # print("conv",x.shape)
        x = self.bn1(x)
       # print("bn",x.shape)
        x = self.relu(x)
        x = self.maxpool(x)
       # print("maxp",x.shape)
        x = self.layer1(x)
       # print("maxp",x.shape)
        x = self.layer2(x)
       # print("maxp",x.shape)
        x = self.layer3(x)
       # print("maxp",x.shape)
        x = self.layer4(x)
       # print("maxp",x.shape)

        x = self.avgpool(x)
        #print('poll:',x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        #return x
def load_data_cross(batch_size=200):
    global global_point
    #print('globale_point: ',global_point)
    data=[]
    label = []
    count = 0
    block_count = 0
    tempblock = []
    tempblock2 = []
    data_id, tempdata,templabel=train_data.__getitem__(global_point)
    pre_id = data_id
    tempblock.append(tempdata)
    tempblock2.append(templabel)
    count = count + 1
    while(1):
        data_id, tempdata,templabel = train_data.__getitem__(global_point)
        if(data_id>pre_id and count<20):#not enough img
            tempblock.append(tempdata)
            tempblock2.append(templabel)
            count = count + 1
           # print(count)
        if(data_id > pre_id and count == 20):# up to 20 img
            data = data + tempblock
            label = label + tempblock2
            block_count = block_count + 1
            tempblock = []
            tempblock2 = []
            count = 0
            if(block_count >= batch_size/20):
                break
        if(count<20 and data_id<pre_id):#another file
            tempblock=[]
            tempblock2 = []
            count = 0
            pre_id = data_id

    data = np.array(data)
    label = np.array(label)

    return torch.from_numpy(data).float(),torch.from_numpy(label).float()
def traindata_fun(epoch):
    model.train()
    criterion = nn.MSELoss(reduce=True,size_average=True)
    #print(train_loader)
    for batch_idx, (data, target) in enumerate(train_loader):
       # print(data.size(),target.size())
        if use_cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx > 2:
            return
        if batch_idx % 50 == 0:
            end = time()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]),'time:',end-start)
            writer.add_scalar('loss/x',loss,batch_idx)
    if epoch%50==0:
        torch.save(model,'./model/STN_gei'+str(epoch)+'.pkl')
#
def train(epoch):
    model.train()
    criterion = nn.MSELoss(reduce=True,size_average=True)
    #print(train_loader)
    #for batch_idx, (data, target) in enumerate(train_loader):
    batch_idx = 0
    batch_size = 100
    ita_num = len(train_loader.dataset)/batch_size
    while(batch_idx < ita_num):
        batch_idx = batch_idx + 1
        #print("epoch:",epoch,"batch-idx: ",batch_idx)
        data,target = load_data_cross(batch_size = batch_size)
       # print(data.size(),target.size())
        if use_cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 5 == 0:
            end = time()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]),'time:',end-start)
            writer.add_scalar('loss/x',loss,batch_idx)
    if epoch%50==0:
        torch.save(model,'./model/STN_gei'+str(epoch)+'.pkl')
#
# 一个简单的测试程序来测量空间转换网络 (STN) 在 MNIST 上的表现.
#


def test():
    model.eval()
    
    test_loss = 0
    correct = 0
    criterion = nn.MSELoss(reduce=True,size_average=True)
    for data, target in test_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)

        # 累加批loss
        test_loss += criterion(output, target, size_average=False).data[0]
        # 得到最大对数几率 (log-probability) 的索引.
        #pred = output.data.max(1, keepdim=True)[1]
        #correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    #print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
    #      .format(test_loss, correct, len(test_loader.dataset),
    #              100. * correct / len(test_loader.dataset)))

def testm():
    model = Net()
    pretrained_file = 'model/STN_gei200.pkl'
    model = torch.load(pretrained_file)
    model.eval()


def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

# 我们想要在训练之后可视化空间转换层 (spatial transformers layer) 的输出, 我们
# 用 STN 可视化一批输入图像和相对于的转换后的数据.


def visualize_stn(num):
    # 得到一批输入数据
    #data, _ = next(iter(test_loader))
    #data, aim = iter(test_loader).next()
    order = 0
    #for data,aim in test_loader:
    batch_idx = 0
    batch_size = 30
    while(batch_idx < test_loader.__len__()/batch_size):
        batch_idx = batch_idx + 1
        data,aim = load_data_cross(batch_size=batch_size)
        data = Variable(data, volatile=True)

        aim = Variable(aim)
        
        if use_cuda:
            data = data.cuda()

        aim_tensor = aim.data
            
        input_tensor = data.cpu().data
        transformed_input_tensor = model.stn(data).cpu().data
        fusion_input_tensor = model.Gei(model.stn(data)).cpu().data
        #print(transformed_input_tensor.size())

        in_grid = convert_image_np(
            torchvision.utils.make_grid(input_tensor,padding = 2))

        aim_grid = convert_image_np(
            torchvision.utils.make_grid(aim_tensor))

        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor))

        fusion_grid = convert_image_np(
            torchvision.utils.make_grid(fusion_input_tensor))
        # 并行地 (side-by-side) 画出结果
        f, axarr = plt.subplots(2, 2)
        axarr[0,0].imshow(in_grid)
        axarr[0,0].set_title('Dataset Images')

        axarr[0,1].imshow(out_grid)
        axarr[0,1].set_title('Transformed Images')

        axarr[1,0].imshow(aim_grid)
        axarr[1,0].set_title('groundtruth ')

        axarr[1,1].imshow(fusion_grid)
        axarr[1,1].set_title('output_fusion')

        plt.savefig('model/'+str(order)+'stn.png')
        plt.show()
        plt.pause(1)
        plt.close()
        order = order+1
        if(order>=num):
            break
    #break
global global_point
global_point = 0
model = Net(BasicBlock,[2,2,2,2])
if use_cuda:
    model.cuda()


optimizer = optim.SGD(model.parameters(), lr=0.01)

writer = SummaryWriter()
start = time()

for epoch in range(1, 200 + 1):
    train(epoch)
    #test()

# 在一些输入批次中可视化空间转换网络 (STN) 的转换
writer.export_scalars_to_json("./log/stn.json")
writer.close()

#testm()


visualize_stn(2)
#plt.ioff()

