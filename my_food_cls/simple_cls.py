import random
import torch
import torch.nn as nn
import numpy as np
import os
import time
import matplotlib.pyplot as plt

from PIL import Image #读取图片数据
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm #显示进度

from torchvision import transforms #用于数据增广
from model_utils.model import initialize_model
# import os
# print(os.getcwd())  # 打印当前工作目录

#设置随机种子，用于固定随机结果
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

#把图片输入大小改成224
HW=224


# 数据增广（训练集的数据变换）
train_transform = transforms.Compose(
    [
        transforms.ToPILImage(), #把图片转换成模型需要的格式
        transforms.RandomResizedCrop(224), #增广：随机放大并裁切
        transforms.RandomRotation(50), #增广：50读以内随机旋转
        transforms.ToTensor()
    ]
)
# 数据增广（验证集的数据变换）
val_transform = transforms.Compose(
    [
        transforms.ToPILImage(), #把图片转换成模型需要的格式

        transforms.ToTensor()
    ]
)

# 1、食物分类数据结构
class food_DataSet(Dataset):
    def __init__(self,path,mode="train"):
        self.mode = mode #在不同函数之间传递mode
        # 半监督学习，只需要读取X
        if mode == "semi": #无标签数据
            self.X = self.read_f(path)
        else: #有标签数据
            self.X,self.Y = self.read_f(path)
            self.Y = torch.LongTensor(self.Y) #标签转成长整型
        if mode == "train": #训练模式
            self.transform = train_transform  #
        else: #验证模式
            self.transform = val_transform
    # 把read_f函数放到类里面,
    # 读入文件
    # def read_f(path):
    def read_f(self,path):
        #半监督学习
        if self.mode == "semi":
            # 列出指定路径下的所有文件，并为这些文件创建一个用于存储图像数据的NumPy数组
            f_list = os.listdir(path)
            xi = np.zeros((len(f_list), HW, HW, 3), dtype=np.uint8)  # dtype=np.unit8指定读出来是整型RGB数据
            # 读取文件夹下面所有照片
            for j, img_name in enumerate(f_list):
                img_path = os.path.join(path, img_name)  # 合并文件路径和图片名字
                img = Image.open(img_path)
                img = img.resize((HW, HW))
                xi[j, ...] = img
            print("读到了%d个数据" % len(xi))
            return xi

        else:
            for i in tqdm(range(11)):  # 遍历每个文件夹
                # 找出文件夹下面所有文件
                f_dir = path + "%02d" % i  # 拼接问题，文件路径要对
                f_list = os.listdir(f_dir)
                # print(f"Processing folder: {f_dir}")
                # print(f"File list: {f_list}")
                # xi是每一类的图片, yi是每一类的标签，长x宽=HWxHW
                xi = np.zeros((len(f_list), HW, HW, 3), dtype=np.uint8)  # dtype=np.unit8指定读出来是整型RGB数据
                yi = np.zeros(len(f_list), dtype=np.uint8)

                # 读取文件夹下面所有照片
                # for img_name in f_list: ##解释一下为什么要改成下面这句
                # for j,img_name in enumerate(tqdm(f_list)):
                for j, img_name in enumerate(f_list):
                    img_path = os.path.join(f_dir, img_name)  # 合并文件路径和图片名字
                    # 读取图片数据
                    img = Image.open(img_path)
                    img = img.resize((HW, HW))
                    # 把图片数据放进数据集
                    xi[j, ...] = img
                    yi[j] = i

                # 把每一类合并成一个大数据集
                if i == 0:
                    X = xi
                    Y = yi
                else:
                    X = np.concatenate((X, xi), axis=0)  # concatenate合并，axis=0纵轴
                    Y = np.concatenate((Y, yi), axis=0)

            print("读到了%d个数据" % len(Y))
            return X, Y

    # 在使用 PyTorch 的 DataLoader 时会被调用，用于在训练和验证过程中逐个获取数据样本
    def __getitem__(self,item): #调试的时候说item没有定义？
        if self.mode == "semi":
            return self.transform(self.X[item]),self.X[item] #返回增广后的样本和原始样本
        else:
            return self.transform(self.X[item]),self.Y[item] #transform(self.X[item])对X作数据增广，返回增广后的样本和对应的标签

    def __len__(self):
        # return len(self.Y)
        return len(self.X)

# 无标签数据集的数据结构，输入为无标签数据集、模型、置信度，输出为加入训练数据集的数据
class semiDataset(Dataset):
    def __init__(self,no_label_loader,model,device,thres=0.99):
        x,y = self.get_label(no_label_loader,model,device,thres) #get_label是模型训练函数

        #对x,y的处理：列表-->张量
        if x==[]:
            self.flag = False #没有超过置信度的数据
        else:
            self.flag = True
            self.X = np.array(x)
            self.Y = torch.LongTensor(y)
            self.transform = train_transform
    def __getitem__(self, item):
        return self.transform(self.X[item]),self.Y[item]

    def __len__(self):
        return len(self.X)

    def get_label(self,no_label_loader,model,device,thres): #一旦有数据要经过模型，就要考虑是否会积攒梯度
        model = model.to(device) #把模型放在设备上
        pred_prob = []
        labels = []
        #保存最终结果
        x=[]
        y=[]

        soft = nn.Softmax()
        # S1让数据通过模型
        with torch.no_grad(): #不计算梯度
            for bat_x,_ in no_label_loader: #取一批数据，bat_x是增广后的数据集，用来承接模型；_是原始数据集，模型训练时用不到
                bat_x = bat_x.to(device) #为什么得到的是（18，224，224，3）不是（16，224，224，3）
                pred = model(bat_x)
                pred_soft = soft(pred)
                pred_max,pred_val = pred_soft.max(1)
                #由于pred_max,pred_val不是一个数，所以不可以用append；也不是列表，所以不能直接用extend，要先放在cpu上面再转成numpy再转成列表
                pred_prob.extend(pred_max.cpu().numpy().tolist())
                labels.extend(pred_val.cpu().numpy().tolist())
        #符合条件的数据加入最终的数据集
        # for prob in pred_prob:
        #     if prob>thres:
        for index,prob in enumerate(pred_prob):
            if prob >thres:
                x.append(no_label_loader.dataset[index][1]) #取原始图片（下标为1），调用到原始的getitem
                y.append(labels[index])
        return x,y

# 存放无标签数据中被打上标签的数据（超过置信度）
def get_semi_loader(no_label_loader,model,device,thres):
    semiset = semiDataset(no_label_loader,model,device,thres)
    if semiset.flag == False: #没有加入新数据
        return None
    else: #加载semi数据
        semi_loader=DataLoader(semiset,batch_size=16,shuffle=False)
        return semi_loader

class cnnLayer(nn.Module):
    def __init__(self,in_cha,out_cha):
        super(cnnLayer,self).__init__()
        self.conv = nn.Conv2d(in_cha,out_cha,3,1,1) #卷积
        self.bn = nn.BatchNorm2d(out_cha) #归一化，通道数out_cha
        self.relu = nn.ReLU() #激活
        self.pool = nn.MaxPool2d(2) #池化，

    def forward(self,x):
        x=self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


# 模型
class myModel(nn.Module):
    # def __init__(self,num_cls): #分类任务要传入分类个数num_cls
    #     super(myModel, self).__init__()
    #     # 3*224*224-->512*7*7-->拉直-->全连接
    #     self.conv1 = nn.Conv2d(3,64,3,1,1) #卷积，64*224*224
    #     self.bn1 = nn.BatchNorm2d(64) #归一化，通道数64
    #     self.relu1 = nn.ReLU()
    #     self.pool1 = nn.MaxPool2d(2) #池化，64*112*112
    #
    #     # 封装成一个层
    #     self.layer1 = nn.Sequential(
    #         nn.Conv2d(64, 128, 3, 1, 1),  # 卷积，128*112*112
    #         nn.BatchNorm2d(128),
    #         nn.ReLU(),
    #         nn.MaxPool2d(2)  # 池化，128*56*56
    #     )
    #
    #     self.layer2 = nn.Sequential(
    #         nn.Conv2d(128, 256, 3, 1, 1),  # 卷积，256*56*56
    #         nn.BatchNorm2d(256),
    #         nn.ReLU(),
    #         nn.MaxPool2d(2)  # 池化，256*28*28
    #     )
    #     self.layer3 = nn.Sequential(
    #         nn.Conv2d(256, 512, 3, 1, 1),  # 卷积，512*28*28
    #         nn.BatchNorm2d(512),
    #         nn.ReLU(),
    #         nn.MaxPool2d(2)  # 池化，512*14*14
    #     )
    #
    #     self.pool2 = nn.MaxPool2d(2) # 池化，512*7*7
    #     self.fc1 = nn.Linear(25088,1000) #25088-->1000
    #     self.relu2 = nn.ReLU()
    #     self.fc2 = nn.Linear(1000,num_cls) #1000-->11

    def __init__(self,num_cls): #分类任务要传入分类个数num_cls
        super(myModel, self).__init__()
        self.layer1 = cnnLayer(3,64)
        self.layer2 = cnnLayer(64, 128)
        self.layer3 = cnnLayer(128, 256)
        self.layer4 = cnnLayer(256, 512)

        self.pool2 = nn.MaxPool2d(2) # 池化，512*7*7
        self.fc1 = nn.Linear(25088,1000) #25088-->1000
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(1000,num_cls) #1000-->11

    def forward(self,x):
        x=self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x=self.pool2(x) #RuntimeError: mat1 and mat2 shapes cannot be multiplied (14336x7 and 25088x1000)也就是说没有拉直，加上下面这行
        x=x.view(x.size()[0],-1)
        x=self.fc1(x)
        x=self.relu2(x)
        x=self.fc2(x)

        return x

# 训练模型（训练+验证）参数：模型，训练集，验证集，设备，轮次，优化器，损失函数，模型保存路径
# def train_val(model,train_loader,val_loader,device,epochs,optimizer,loss,save_path):
def train_val(model, train_loader, val_loader, nolabel_loader, device, epochs, optimizer, loss, thres, save_path):
    model = model.to(device)

    #记录loss，分类问题还需要记录acc
    plt_train_loss = [] #保存所有轮次的loss
    plt_val_loss = [] #plt提醒待会要画图

    plt_train_acc = []
    plt_val_acc = []
    # min_val_loss =9999999999 #记录最优模型的loss
    max_acc = 0.0 #分类问题使用准确率作为保存模型的根据

    #开始训练
    for epoch in range(epochs):
        semi_loader = None
        train_loss = 0.0
        val_loss = 0.0
        train_acc = 0.0
        val_acc = 0.0

        # 半监督学习的loss和acc
        semi_loss =0.0
        semi_acc = 0.0

        start_t = time.time()

        model.train() #进入训练模式
        for batch_x,batch_y in train_loader: #在训练集中取一批数据
            x,target = batch_x.to(device),batch_y.to(device) #放在GPU上面训练
            pred = model(x) #得到预测值
            # print("pred shape:", pred.shape)  # 应该是 (batch_size, num_classes)
            # print("target shape:", target.shape)  # 应该是 (batch_size,)
            train_bat_loss = loss(pred, target) # 9、优化：损失函数正则化
            train_bat_loss.backward() #梯度回传
            optimizer.step() #更新模型
            optimizer.zero_grad() #重置loss值
            train_loss += train_bat_loss.cpu().item() #把train_bat_loss放在cpu上面（才能和train_loss相加）取数值
            train_acc += np.sum(np.argmax(pred.detach().cpu().numpy(),axis=1) == target.cpu().numpy()) #每一轮预测对的数量累加

        plt_train_loss.append(train_loss / train_loader.__len__()) #将本轮计算的loss加到已有的plt_train_loss上面并除以总轮数
        plt_train_acc.append(train_acc / train_loader.dataset.__len__())  #记录准确率：预测对的个数/数据集总长度

        if semi_loader != None:
            for batch_x,batch_y in semi_loader: #在训练集中取一批数据
                x,target = batch_x.to(device),batch_y.to(device) #放在GPU上面训练
                pred = model(x) #得到预测值
                semi_bat_loss = loss(pred, target) # 9、优化：损失函数正则化
                semi_bat_loss.backward() #梯度回传
                optimizer.step() #更新模型
                optimizer.zero_grad() #重置loss值
                semi_loss += train_bat_loss.cpu().item() #把train_bat_loss放在cpu上面（才能和train_loss相加）取数值
                semi_acc += np.sum(np.argmax(pred.detach().cpu().numpy(),axis=1) == target.cpu().numpy()) #每一轮预测对的数量累加
            print("半监督数据集训练准确率：",semi_acc / train_loader.dataset.__len__())

        model.eval() #进入验证模式
        with torch.no_grad(): #不记录梯度
            for batch_x, batch_y in val_loader:
                x, target = batch_x.to(device), batch_y.to(device)
                pred = model(x)
                val_bat_loss = loss(pred, target)  # 9、优化：损失函数正则化
                val_loss += val_bat_loss.cpu().item()  # 把train_bat_loss放在cpu上面（才能和train_loss相加）取数值
                val_acc += np.sum(
                    np.argmax(pred.detach().cpu().numpy(), axis=1) == target.cpu().numpy())

        plt_val_loss.append(val_loss / val_loader.__len__())  # 将本轮计算的loss加到已有的plt_train_loss上面并除以总轮数
        plt_val_acc.append(val_acc / val_loader.dataset.__len__())

        # #在验证之后做判断，实际应用中，如果准确率达到0.7，才生成semiLoader
        # if plt_val_acc[-1] > 0.7:
        #     semi_loader = get_semi_loader(nolabel_loader,model,device,thres)
        # 每过5轮读取一次semiloader
        if epoch%5 ==0 and plt_val_acc[-1] > 0.7:
            semi_loader = get_semi_loader(nolabel_loader, model, device, thres)

        # 保存模型
        if val_acc > max_acc:
            torch.save(model,save_path)
            max_acc = val_acc

        # 打印这一轮的训练结果
        print("[%03d/%03d]  %2.2f sec(s) Trainloss:%.6f | Valloss:%.6f; Trainacc:%.6f | Valacc:%.6f"% \
              (epoch,epochs,time.time() - start_t,plt_train_loss[-1],plt_val_loss[-1],plt_train_acc[-1],plt_val_acc[-1]))

        #画一下图(可以封装)
        plt.plot(plt_train_loss)
        plt.plot(plt_val_loss)
        plt.title("loss")
        plt.legend(["train","val"])
        plt.show()
        # 指定图片保存路径
        figure_save_path = "pic/loss/"  # 先创建一个目录
        if not os.path.exists(figure_save_path):
            os.makedirs(figure_save_path)  # 如果不存在目录，则创建
        plt.savefig(os.path.join(figure_save_path, str(epoch)))  # 分别命名图片

        plt.plot(plt_train_acc)
        plt.plot(plt_val_acc)
        plt.title("acc")
        plt.legend(["train", "val"])
        plt.show()
        # 指定图片保存路径
        figure_save_path = "pic/acc/"  # 先创建一个目录
        if not os.path.exists(figure_save_path):
            os.makedirs(figure_save_path)  # 如果不存在目录，则创建
        plt.savefig(os.path.join(figure_save_path, str(epoch)))  # 分别命名图片

#1、读文件
# 1 确认路径是否完整：用于指定数据集的路径
# train_path = "food-11/training/labeled/" #拼接问题，文件路径要对，第33行打断点调试
# val_path = "food-11/validation/"
train_path = "food-11_sample/training/labeled/" #FileNotFoundError: [WinError 3] 系统找不到指定的路径。: 'food-11_sample/training/labeled00'拼接问题，文件路径要对，第33行打断点调试 #训练集的路径，包含有标签的训练图像。
val_path = "food-11_sample/validation/"
nolabel_path = "food-11_sample/training/unlabeled/00"
#2、传入数据集
train_set = food_DataSet(train_path,"train")
val_set = food_DataSet(val_path,"val")
nolabel_set = food_DataSet(nolabel_path,"semi")

train_loader = DataLoader(train_set,batch_size=16,shuffle=True) #shuffle=True随机打乱
val_loader = DataLoader(val_set,batch_size=16,shuffle=True)
nolabel_loader = DataLoader(nolabel_set,batch_size=16,shuffle=False) #打标签不能打乱
# semidata_loader =

#初始化模型
# model = myModel(11)
# # 用现成模型
# from torchvision.models import resnet18
# model = resnet18(pretrained=True) #pretrained=True表示用训练过的参数
# in_features = model.fc.in_features #提取输入特征维度，替换分类头
# model.fc = nn.Linear(in_features,11)
model,_ = initialize_model("resnet18",11,use_pretrained=True)

lr = 0.001 #学习率
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
device = "cuda" if torch.cuda.is_available() else "cpu"
save_path = "my_food_cls/model_save/best_model.pth"
epochs = 20
thres = 0.99 #实际应用置信度一定要高

# semiset = semiDataset(nolabel_loader,model,device,thres = 0.99)


# 训练并验证
# train_val(model,train_loader,val_loader,device,epochs,optimizer,loss,save_path)
train_val(model,train_loader,val_loader,nolabel_loader,device,epochs,optimizer,loss,thres,save_path)


# for batch_x,batch_y in train_loader:
#     pred = model(batch_x)



