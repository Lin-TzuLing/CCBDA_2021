import torchvision.transforms as transforms
from PIL import Image
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import numpy as np
from torch.utils.data import DataLoader


# 讀取train image檔案，dir設定為train data的位置
dir = './unlabeled'
data = []
for file in os.listdir(dir):
    if file.endswith('.jpg'):
        # 讀取個別image
        file = dir+'/'+file
        data.append(Image.open(file))
        continue
    else:
        print("found non-jpg file in dir")
        continue
del dir, file


# 將圖片進行轉換，隨機安排轉換的方式
transform1 = transforms.Compose(
    [transforms.RandomHorizontalFlip(p=0.8),
    transforms.ColorJitter(brightness=(1, 2), contrast=(1, 4), saturation=(1, 5), hue=(0.2, 0.4)),
     ]
)

transform2 = transforms.Compose(
    [transforms.RandomHorizontalFlip(p=0.2),
    transforms.Grayscale(num_output_channels=3),
    transforms.CenterCrop((84,84)),
     ]
)


transform_resize = transforms.Resize((96,96))
convert_tensor = transforms.ToTensor()



now = datetime.now()
print('transform training data start : {}'.format(now.strftime("%H:%M:%S")))

original_data, trans_data1, trans_data2 = [], [], []
for image in data:
    # 原圖的tensor
    original_image = convert_tensor(image)
    original_image = torch.unsqueeze(original_image, dim=0)
    original_data.append(original_image)
    # 轉換後的圖_1
    trans_image1 = transform_resize(transform1(image))
    trans_image1 = convert_tensor(trans_image1)
    trans_image1 = torch.unsqueeze(trans_image1, dim=0)
    trans_data1.append(trans_image1)
    # 轉換後的圖_2
    trans_image2 = transform_resize(transform2(image))
    trans_image2 = convert_tensor(trans_image2)
    trans_image2 = torch.unsqueeze(trans_image2, dim=0)
    trans_data2.append(trans_image2)
original_data = torch.cat(original_data)
trans_data1 = torch.cat(trans_data1)
trans_data2 = torch.cat(trans_data2)

now = datetime.now()
print('transform training data end : {}'.format(now.strftime("%H:%M:%S")))
del image, original_image, trans_image1, trans_image2



# 讀取test的image檔案，dir設定為test data的位置
directory = './test'
test_data, class_label = [], []
for label in os.listdir(directory):
    sub_dir = directory+'/'+label
    for file in os.listdir(sub_dir):
        if file.endswith('.jpg'):
            # 讀取個別image
            file = sub_dir+'/'+file
            image = convert_tensor(Image.open(file))
            image = torch.unsqueeze(image, dim=0)
            test_data.append(image)
            # image class type
            class_label.append(int(label))
            continue
        else:
            print("not jpg")
            continue
test_data = torch.cat(test_data)
class_label = torch.Tensor(class_label)
del directory, file

print('原圖 : {}'.format(original_data.size()))
print('train_轉換圖1 : {}'.format(trans_data1.size()))
print('train_轉換圖2 : {}'.format(trans_data2.size()))
print('test_data : {}'.format(test_data.size()))
print('class_label : {}'.format(class_label.size()))



# similarity loss
def xt_xent(u, v, temperature=0.5):
    N = u.shape[0]
    z = torch.cat([u, v], dim=0)
    z = F.normalize(z, p=2, dim=1)

    # temperature => embedding uniformity
    s = torch.matmul(z, z.t()) / temperature

    # 對角線=1的matrix (torch.eye)
    mask = torch.eye(2 * N).bool().to(z.device)

    # mask中1(true)的地方，s同樣index的地方要替換成value -inf
    # 原圖和轉換圖互為label    
    s = torch.masked_fill(s, mask, -float('inf'))
    label = torch.cat([
        torch.arange(N, 2 * N),
        torch.arange(N)]).to(z.device)

    loss = F.cross_entropy(s, label)
    return loss


def KNN(emb, cls, batch_size, Ks=[1, 10, 50, 100]):
    """Apply KNN for different K and return the maximum acc"""
    preds = []
    mask = torch.eye(batch_size).bool().to(emb.device)
    mask = F.pad(mask, (0, len(emb) - batch_size))
    for batch_x in torch.split(emb, batch_size):
        dist = torch.norm(
            batch_x.unsqueeze(1) - emb.unsqueeze(0), dim=2, p="fro")
        now_batch_size = len(batch_x)
        mask = mask[:now_batch_size]
        dist = torch.masked_fill(dist, mask, float('inf'))
        # update mask
        mask = F.pad(mask[:, :-now_batch_size], (now_batch_size, 0))
        pred = []
        for K in Ks:
            knn = dist.topk(K, dim=1, largest=False).indices
            knn = cls[knn].cpu()
            pred.append(torch.mode(knn).values)
        pred = torch.stack(pred, dim=0)
        preds.append(pred)
    preds = torch.cat(preds, dim=1)
    accs = [(pred == cls.cpu()).float().mean().item() for pred in preds]
    return max(accs)



class myModel(nn.Module):
    def __init__(self):
        super(myModel, self).__init__()
        layer = []

        # block 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=5, padding_mode='zeros', padding='same')
        self.bn1 = nn.BatchNorm2d(num_features=4, affine=True)
        self.relu1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding_mode='zeros', padding='same')
        self.bn1_2 = nn.BatchNorm2d(num_features=4, affine=True)
        self.relu1_2 = nn.ReLU()
        self.pool1 = nn.AdaptiveAvgPool2d(64)

        # block 2
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=5, padding_mode='zeros', padding='same')
        self.bn2 = nn.BatchNorm2d(num_features=8, affine=True)
        self.relu2 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding_mode='zeros', padding='same')
        self.bn2_2 = nn.BatchNorm2d(num_features=8, affine=True)
        self.relu2_2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(kernel_size=2)


        # block 3
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding_mode='zeros', padding='same')
        self.bn3 = nn.BatchNorm2d(num_features=8, affine=True)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.AvgPool2d(kernel_size=2)

        # block 4
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding_mode='zeros', padding='same')
        self.bn4 = nn.BatchNorm2d(num_features=16, affine=True)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.AdaptiveAvgPool2d(8)


        # linear
        self.linear= nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=1024, out_features=800, bias=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=800, out_features=512, bias=True),
        )

        # projection
        self.projection = nn.Sequential(
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=32),
        )

        layer += [self.conv1, self.bn1, self.relu1, self.pool1]
        layer += [self.conv2, self.bn2, self.relu2, self.pool2]
        layer += [self.conv3, self.bn3, self.relu3]
        layer += [self.conv4, self.bn4, self.relu4, self.pool4]
        self.conv = nn.Sequential(*layer)

    def forward(self, x, flag_train):
        out = self.conv(x)
        out = self.linear(out)
        if flag_train==True:
            out = self.projection(out)
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


model = myModel()
# initialize weight, bias of layer in model
model.initialize_weights()


def predEmbed():
    model.eval()
    with torch.no_grad():
        pred_embed = []
        x = DataLoader(original_data, batch_size=200)
        for _, data in enumerate(x):
            # 預測unlabeled data的embedding
            pred_embed.append(model(data, flag_train=False))
        pred_embed = torch.cat(pred_embed)
    return pred_embed



def testAcc():
    model.eval()
    with torch.no_grad():
        # trans_testData通過model得到test data embedding
        test_embedding = model(test_data, flag_train=False)
        acc = KNN(test_embedding, class_label, batch_size=500)
    return acc

def l1_regularizer(model, lambda_l1=0.01):
    lossl1 = 0
    for model_param_name, model_param_value in model.named_parameters():
            if model_param_name.endswith('weight'):
                lossl1 += lambda_l1 * model_param_value.abs().sum()
    return lossl1

def trainModel(epochs,batch_size,):
    # loss and optimizer
    learning_rate = 0.00005
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    best_acc = 0
    
    # batch     
    x1 = DataLoader(trans_data1, batch_size=batch_size)
    x2 = DataLoader(trans_data2, batch_size=batch_size)

    # training loop
    print('start training')

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        # batch training     
        for i, (data1, data2) in enumerate(zip(x1, x2)):
            encoded_x1 = model(data1, flag_train=True)
            encoded_x2 = model(data2, flag_train=True)
            optimizer.zero_grad()
            # similarity loss, temperature可從0~1
            loss = xt_xent(encoded_x1,encoded_x2, temperature=0.5)
            # l2 loss
            # l2_lambda = 0.1
            # l2_lambda = 0.5
            l2_norm_model = sum(p.pow(2.0).sum()
                          for p in model.parameters())
            # loss = loss + l2_lambda * l2_norm_model
            loss = loss + l1_regularizer(model, 0.08)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # train 1 epoch test 1 次         
        acc = testAcc()
        print('epoch {}, loss {}, acc {}'.format(epoch+1,running_loss/i,acc))
        if acc > best_acc:
            pred_embed = predEmbed()
            best_acc = acc
            print('pred embed at epoch {}'.format(epoch+1))
    # 回傳最好的那個pred embed
    return pred_embed


if __name__ == "__main__":
    pred_embed = trainModel(epochs=30, batch_size=512)
    print('Finished Training')
    pred_embed = pred_embed.detach().numpy()
    with open('./0712118.npy', 'wb') as file:
        np.save(file, pred_embed)

print('done')




