import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from p3d_model import P3D,Bottleneck
import numpy as np
import torch.nn as nn
import torch.optim as optim
from time  import time
import visdom
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import visdom
import random
def get_optim_policies(model=None,modality='Flow',enable_pbn=True):
    #print('this is not used/!#################################')
    '''''
    first conv:         weight --> conv weight
                        bias   --> conv bias
    normal action:      weight --> non-first conv + fc weight
                        bias   --> non-first conv + fc bias
    bn:                 the first bn2, and many all bn3.

    '''
    first_conv_weight = []
    first_conv_bias = []
    normal_weight = []
    normal_bias = []
    bn = []

    if model==None:
        log.l.info('no model!')
        exit()

    conv_cnt = 0
    bn_cnt = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Conv3d) or isinstance(m, torch.nn.Conv2d):
            ps = list(m.parameters())
            conv_cnt += 1
            if conv_cnt == 1:
                first_conv_weight.append(ps[0])
                if len(ps) == 2:
                    first_conv_bias.append(ps[1])
            else:
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
        elif isinstance(m, torch.nn.Linear):
            ps = list(m.parameters())
            normal_weight.append(ps[0])
            if len(ps) == 2:
                normal_bias.append(ps[1])
              
        elif isinstance(m, torch.nn.BatchNorm3d):
            bn_cnt += 1
            # later BN's are frozen
            if not enable_pbn or bn_cnt == 1:
                bn.extend(list(m.parameters()))
        elif isinstance(m,torch.nn.BatchNorm2d):
            bn.extend(list(m.parameters()))
        elif len(m._modules) == 0:
            if len(list(m.parameters())) > 0:
                raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

    slow_rate=0.7
    n_fore=int(len(normal_weight)*slow_rate)
    slow_feat=normal_weight[:n_fore] # finetune slowly.
    slow_bias=normal_bias[:n_fore] 
    normal_feat=normal_weight[n_fore:]
    normal_bias=normal_bias[n_fore:]

    return [
        {'params': first_conv_weight, 'lr_mult': 5 if modality == 'Flow' else 1, 'decay_mult': 1,
         'name': "first_conv_weight"},
        {'params': first_conv_bias, 'lr_mult': 10 if modality == 'Flow' else 2, 'decay_mult': 0,
         'name': "first_conv_bias"},
        {'params': slow_feat, 'lr_mult': 1, 'decay_mult': 1, 'name': "slow_feat"},
        {'params': slow_bias, 'lr_mult': 2, 'decay_mult': 0,'name': "slow_bias"},
        {'params': normal_feat,'decay_mult': 1,'lr':0.01,'name': "normal_feat"},
        {'params': normal_bias,  'decay_mult':1, 'lr':0.01,'name': "normal_bias"},
        {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
         'name': "BN scale/shift"},
        #{'params':model.myfc.parameters(), 'lr': 0.001},
    ]

root = '/home/xiaoqian/pseudo-3d-pytorch/'
# -----------------ready the dataset--------------------------
def default_loader(path):
    return Image.open(path)
class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],int(words[1])))
        self.imgs = imgs
        

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):

        #fn, label = self.imgs[index]

        #if self.transform is not None:
        #    img = self.transform(img)
        global global_point 
        index = int(index%(self.__len__()))
        if index+16>=self.__len__():# file end 
            index = 0
        #print('index: ',index,'len: ',self.__len__())
        xlist = []
        ylist = []
       
        blocklist  = []
        while(1):# define the index 
            imgname1 = self.imgs[index][0]
            imgname2 = self.imgs[index+16][0]
            if imgname1[:-7] != imgname2[:-7]:
                oldfile = imgname1[-7]
                for i in range(index, index+16):
                    imgnamei = self.imgs[index][0][:-7]
                    if imgnamei!=oldfile:
                        index = i
            if index+16>=self.__len__():# file end 
                index = 0
            if self.imgs[index][0][:-7] == self.imgs[index+16][0][:-7]:
                break

        for i in range(index, index+16):
            fnx, label = self.imgs[i]
            #print(fn)
            locate = '/mnt/disk50/datasets/dataset-gait/CASIA-B-flow/y/'
            fny = locate+fnx[len(locate):]
            imgx = self.loader(fnx)# x 224*224
            imgy =self.loader(fny)# y
            #img = img1+img2
            #img = self.loader(fn)  


            width = imgx.size[0] #640
            height= imgx.size[1] #480
            ratio_h = 160/height
            ratio_w = 160/width
            #img_data = np.array(img)
           # print(width,height)
            plt.ion()
            #new_img = np.zeros((160,160,3),np.uint8)
            if ratio_h>ratio_w: #width>height
                ratio =ratio_h
                imgx = imgx.resize((int(width*ratio),160),Image.ANTIALIAS)
                img_datax = np.array(imgx)

                #print("###",img_data.shape)
                #plt.imshow(img_data)
                new_width = int(width*ratio)
                new_height = 160
                img_datax = img_datax[:160,int(new_width/2)-80:int(new_width/2)+80]

                imgy = imgy.resize((int(width*ratio),160),Image.ANTIALIAS)
                img_datay = np.array(imgy)
                #print("###",img_data.shape)
                #plt.imshow(img_data)
                new_width = int(width*ratio)
                new_height = 160
                img_datay = img_datay[:160,int(new_width/2)-80:int(new_width/2)+80] 

            else:#height>width
                ratio =ratio_w
                imgx = imgx.resize((160,int(height*ratio)),Image.ANTIALIAS)
                #print(img.size)
                img_datax = np.array(imgx)
                new_height = height*ratio
                img_datax = img_datax[int(new_height/2)-800:int(new_height/2)+80,:160]

                ratio =ratio_w
                imgy = imgy.resize((160,int(height*ratio)),Image.ANTIALIAS)
                #print(img.size)
                img_datay = np.array(imgy)
                new_height = height*ratio
                img_datay = img_datay[int(new_height/2)-80:int(new_height/2)+80,:160]               # for x in range(int(width*ratio)):
               #     for y in range(160):
               #         new_img[y,x,:] = img_data[y,x,:]
            #img2 = Image.fromarray(new_img)
            '''
            print(img_datax,img_datay)
            plt.imshow(img_datax)
            plt.show()
            plt.pause(1)
            plt.close()
            '''
           
           
           # img_data = new_img
            #img_data = np.array(img)
            #img_data = np.array(img)
            #img_data = train_data.__getitem__(i)[0]
            img_x = img_datax
            img_y = img_datay
            
            #print(img_r.shape)
            xlist.append(img_x)
            ylist.append(img_y)
            
        global_point = index + 10
        xarray = np.array(xlist)
        blocklist.append(xarray)
        yarray = np.array(ylist)
        blocklist.append(yarray)

        #print(np.array(blocklist).shape)
        train_block = np.array(blocklist)
        #trainset = []#16 480 640 3
        #np.transpose(train_block,(3,16,480,640))
        #trainset.append(train_block)
        #trainT =  torch.from_numpy(np.array(trainset)).float()
        #print(trainT.shape)
        #return np.array(img),label
        labellist = np.zeros(101)
        intlabel = int(label)
        labellist[label] = 1

        return train_block, labellist,label
        #return fn, label

    def __len__(self):
        return len(self.imgs)

train_data=MyDataset(txt=root+'flowtrainlist.txt', transform=transforms.ToTensor())
#print(img1[0].shape,torch.tensor(img1[0]))
test_data=MyDataset(txt=root+'flowtestlist.txt', transform=transforms.ToTensor())
def load_data(batch_size=10):
    global global_point
    
    #print('globale_point: ',global_point)
    data=[]
    label = []
    for batchpt in range(batch_size):
        tempdata,templabel,templabel2 = train_data.__getitem__(global_point)
        #data.append(train_data.__getitem__(global_point+batchpt)[0])
        #label.append(train_data.__getitem__(global_point+batchpt)[1])
        data.append(tempdata)
        label.append(templabel)
        #print('batch',b,'global_point')
    #global_point = global_point + batch_size    
    return torch.from_numpy(np.array(data)).float(),torch.from_numpy(np.array(label)).float()
def load_data_cross(batch_size=10):
    global global_point
    #print('globale_point: ',global_point)
    data=[]
    label = []
    for batchpt in range(batch_size):
        #data.append(train_data.__get.item__(global_point+batchpt)[0])
        #label.append(train_data.__getitem__(global_point+batchpt)[2])
        tempdata,templabel,templabel2 = train_data.__getitem__(global_point)
        data.append(tempdata)
        label.append(templabel2)
        #print('batchpt',batchpt,'global_point:',global_point)
    #global_point = global_point + batch_size
    return torch.from_numpy(np.array(data)).float(),torch.from_numpy(np.array(label)).long()

def load_data_test(batch_size=10):
    global global_point
    #print('global_point: ',global_point)

    data=[]
    label = []
    for batchpt in range(batch_size):
        tempdata,templabel,templabel2 = train_data.__getitem__(global_point)
        data.append(tempdata)
        label.append(templabel2)
        #print('batchpt:',batchpt,'global_point:',global_point)
    return torch.from_numpy(np.array(data)).float(), np.array(label)

pretrained_file = './model/p3d_flow_199.checkpoint.pth.tar'
#predic = premodel.state_dict()
num_classes = 101
model = P3D(Bottleneck, [3, 8, 36, 3],num_classes=num_classes,modality='Flow')
#model = P3D(Bottleneck,[3,8,36,3],modality='RGB')
model_dict = model.state_dict()
weights=torch.load(pretrained_file)['state_dict']
#print(weights)

#model.load_state_dict(weights)
weight2 = {k:v for k,v in weights.items() if k in model_dict}
model_dict.update(weight2)
#model = torch.load(pretrained_file)
#weight2 = {k:v for k,v in weights.items() if k in model_dict}
#model_dict.update(weight2)
#model.load_state_dict(weights)
#model.load_state_dict(model_dict)


#criterion = nn.MSELoss()# float tensor cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.0001,momentum=0.9)
#optimizer = optim.SGD(get_optim_policies(model = model),lr=0.00001,momentum=0.9)
#print(model)
global_point = 0
print('model ok')
model = model.cuda()
logfile = './log/flow_pre199.txt'
f = open(logfile,'w')
start = time()
countall = 0
batch_size = 15
writer = SummaryWriter()

for ita in range(9010):
    #print("###")
    data1,label1 = load_data_cross(batch_size=batch_size) #5
   # print(label1)
    data = torch.autograd.Variable(data1.cuda())
    optimizer.zero_grad()
    out = model(data)

    #countall = 0
        #print('out#############\n',out)
    tout = out.cpu().detach().numpy()
    #print(out.shape,label2.shape)
    pred = []
    maxp = -1
    max_index = 0
    for i in range(batch_size):
        one_out = tout[i,:]
        for j in range(num_classes):
            if one_out[j]>maxp:
                max_index = j
                maxp = one_out[j]
        pred.append(max_index)
    pred = np.array(pred)
    #print(pred.shape)
    #pred label
    label2=np.array(label1)
    static = pred-label2
    #print(static)
    zero = static==0
    zero_num = static[zero]
    #print(zero_num.size)
    countall = countall + zero_num.size
    #accuracy = countall/(batch_size*1.0)


    loss = criterion(out,label1.cuda())
    loss.backward()
    optimizer.step()
    end = time()
    print("ita:",ita," loss:",str(loss.cpu().detach().numpy())," time: ",str(int(end-start)))
    if(ita%100==0):
        
        accuracy = countall/(100.0*batch_size)
        writer.add_scalar('loss/x',loss,ita)
        writer.add_scalar('accuracy/x',accuracy,ita)
        print("ita:",ita," loss:",str(loss.cpu().detach().numpy()),"accuracy: "+ str(accuracy)+ " time: ",str(int(end-start)))
        f.write(str(ita)+' ' + str(loss.cpu().detach().numpy())+' '+str(accuracy)+'\n')
        countall=0
    if(ita>=2000 and ita%500==0):
        torch.save(model,'./model/flow_pre199_'+str(ita)+'.pkl')

    #print(out.size(),out)
writer.export_scalars_to_json("./log/flow_pre199.json")
writer.close()
'''
parameters = get_optim_policies(model=model,modality='RGB',enable_pbn=True)


data=torch.autograd.Variable(torch.rand(1,3,16,240,160)).cuda()
out=model(data)
print (out.size(),out)
'''