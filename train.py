# 导入包
import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from model import AlexNet
import os
import json
import time

# 使用GPU训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open(os.path.join("train.log"), "a") as log:
    log.write(str(device)+"\n")

#数据预处理
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),       # 随机裁剪，再缩放成 224×224
                                 transforms.RandomHorizontalFlip(p=0.5),  # 水平方向随机翻转，概率为 0.5, 即一半的概率翻转, 一半的概率不翻转
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),

    "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

#导入、加载 训练集
# 导入训练集
#train_set = torchvision.datasets.CIFAR10(root='./data', 	 # 数据集存放目录
#										 train=True,		 # 表示是数据集中的训练集
#                                        download=True,  	 # 第一次运行时为True，下载数据集，下载完成后改为False
#                                        transform=transform) # 预处理过程
# 加载训练集                              
#train_loader = torch.utils.data.DataLoader(train_set, 	  # 导入的训练集
#										   batch_size=50, # 每批训练的样本数
#                                          shuffle=False,  # 是否打乱训练集
#                                          num_workers=0)  # num_workers在windows下设置为0


# 获取图像数据集的路径
data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  		# get data root path 返回上上层目录
image_path = data_root + "花卉识别系统beta/flower_data/"  				 		# flower data_set path

# 导入训练集并进行预处理
train_dataset = datasets.ImageFolder(root=image_path + "/train",		
                                     transform=data_transform["train"])
train_num = len(train_dataset)

# 按batch_size分批次加载训练集
train_loader = torch.utils.data.DataLoader(train_dataset,	# 导入的训练集
                                           batch_size=32, 	# 每批训练的样本数
                                           shuffle=True,	# 是否打乱训练集
                                           num_workers=0)	# 使用线程数，在windows下设置为0

#导入、加载 验证集
# 导入验证集并进行预处理
validate_dataset = datasets.ImageFolder(root=image_path + "/val",
                                        transform=data_transform["val"])
val_num = len(validate_dataset)

# 加载验证集
validate_loader = torch.utils.data.DataLoader(validate_dataset,	# 导入的验证集
                                              batch_size=32, 
                                              shuffle=True,
                                              num_workers=0)

# 存储 索引：标签 的字典
# 字典，类别：索引 {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
flower_list = train_dataset.class_to_idx
# 将 flower_list 中的 key 和 val 调换位置
cla_dict = dict((val, key) for key, val in flower_list.items())

# 将 cla_dict 写入 json 文件中
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

#训练过程
net = AlexNet(num_classes=5, init_weights=True)  	  # 实例化网络（输出类型为5，初始化权重）
net.to(device)									 	  # 分配网络到指定的设备（GPU/CPU）训练
loss_function = nn.CrossEntropyLoss()			 	  # 交叉熵损失
optimizer = optim.Adam(net.parameters(), lr=0.0002)	  # 优化器（训练参数，学习率）

save_path = './AlexNet.pth'
best_acc = 0.0

for epoch in range(150):
    ########################################## train ###############################################
    net.train()     					# 训练过程中开启 Dropout
    running_loss = 0.0					# 每个 epoch 都会对 running_loss  清零
    time_start = time.perf_counter()	# 对训练一个 epoch 计时
    
    for step, data in enumerate(train_loader, start=0):  # 遍历训练集，step从0开始计算
        images, labels = data   # 获取训练集的图像和标签
        optimizer.zero_grad()	# 清除历史梯度
        
        outputs = net(images.to(device))				 # 正向传播
        loss = loss_function(outputs, labels.to(device)) # 计算损失
        loss.backward()								     # 反向传播
        optimizer.step()								 # 优化器更新参数
        running_loss += loss.item()
        
        # 打印训练进度（使训练过程可视化）
        rate = (step + 1) / len(train_loader)           # 当前进度 = 当前step / 训练一轮epoch所需总step
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        with open(os.path.join("train.log"), "a") as log:
              log.write(str("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss))+"\n")
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
    print()
    with open(os.path.join("train.log"), "a") as log:
              log.write(str('%f s' % (time.perf_counter()-time_start))+"\n")
    print('%f s' % (time.perf_counter()-time_start))

    ########################################### validate ###########################################
    net.eval()    # 验证过程中关闭 Dropout
    acc = 0.0  
    with torch.no_grad():
        for val_data in validate_loader:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]  # 以output中值最大位置对应的索引（标签）作为预测输出
            acc += (predict_y == val_labels.to(device)).sum().item()    
        val_accurate = acc / val_num
        
        # 保存准确率最高的那次网络参数
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
        with open(os.path.join("train.log"), "a") as log:
              log.write(str('[epoch %d] train_loss: %.3f  test_accuracy: %.3f \n' %
              (epoch + 1, running_loss / step, val_accurate))+"\n")
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f \n' %
              (epoch + 1, running_loss / step, val_accurate))
with open(os.path.join("train.log"), "a") as log:
      log.write(str('Finished Training')+"\n")
print('Finished Training')
