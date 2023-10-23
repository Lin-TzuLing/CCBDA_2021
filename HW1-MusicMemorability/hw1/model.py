import numpy as np
import os
import torch
import pandas as pd
from sklearn.linear_model import LinearRegression
# import tensorflow as tf
import torch.nn as nn
from torch.optim import optimizer
import torch.nn.functional as F
from argparse import ArgumentParser

# $ python test.py --filename [list of audio file path to predict]
parser = ArgumentParser()
parser.add_argument('--filename', dest='predict_list', type=str, nargs='+', help='input list of audio file path to predict')
predict_list = parser.parse_args().predict_list
# print(predict_list)

# read train feature (trainX)
store = torch.load('./processed_data/train/train_x.pth')
audio_dict, trainX_country, trainX_views = store['audio_dict'], store['country'], store['views']
trainX_stft, trainX_db, trainX_zeroCrossRate = store['f1_stft'], store['f2_db'], store['f3_zeroCrossRate']
trainX_centroid, trainX_rolloff, trainX_mfcc  = store['f4_centroid'], store['f5_rolloff'], store['f6_mfcc']
trainX_bandwidth, trainX_contrast, trainX_flat = store['f7_bandwidth'], store['f8_contrast'], store['f9_flat']
trainX_tempo, trainX_beat, trainX_chroma = store['f10_tempo'], store['f11_beat'], store['f12_chroma']

# read train score (trainY)
train_y = pd.read_csv('processed_data/train/train_y.csv', index_col=0)
train_y = torch.tensor(train_y['score'])

# read test feature (testX)
store = torch.load('./processed_data/test/test_x.pth')
testX_country, testX_views, testX_stft, testX_db = store['country'], store['views'], store['f1_stft'], store['f2_db']
testX_zeroCrossRate, testX_centroid, testX_rolloff = store['f3_zeroCrossRate'], store['f4_centroid'], store['f5_rolloff']
testX_mfcc, testX_bandwidth, testX_contrast = store['f6_mfcc'], store['f7_bandwidth'], store['f8_contrast']
testX_flat, testX_tempo, testX_beat, testX_chroma = store['f9_flat'], store['f10_tempo'], store['f11_beat'], store['f12_chroma']
test = pd.read_csv('processed_data/test/test_x.csv', index_col=0)
del store


# Linear Regression
model_1 = LinearRegression()
train_x = torch.cat([trainX_views,trainX_zeroCrossRate],dim=1)
test_x = torch.cat([testX_views,testX_zeroCrossRate],dim=1)
model_1.fit(train_x, train_y)
y_pred1 = model_1.predict(test_x)
y_pred1 = np.expand_dims(y_pred1, axis=1)
# # 寫入csv
y_pred = pd.DataFrame(y_pred1)
y_pred.columns = ['score']
ans1= pd.DataFrame()
audio_name = test.drop(test.columns.difference(['track']), axis=1)
ans1 = pd.concat([audio_name, y_pred], axis=1)
ans1.to_csv('./result/result.csv',index=False)
# print()


## 試看看把2d的freq圖用cnn分析，用rnn的變形(gru,lstm)，用attention
# 2. 用看看cnn/dnn/rnn
# 1. model
# trainX_db = torch.unsqueeze(torch.mean(torch.mean(trainX_db,dim=2),dim=1),dim=1)
# testX_db = torch.unsqueeze(torch.mean(torch.mean(testX_db,dim=2),dim=1),dim=1)
trainX_contrast = torch.mean(trainX_contrast,dim=1)
testX_contrast = torch.mean(testX_contrast,dim=1)
x1_train = torch.cat([trainX_centroid,trainX_rolloff],dim=1)
x1_test = torch.cat([testX_centroid,testX_rolloff],dim=1)
x2_train = trainX_bandwidth
x2_test = testX_bandwidth
# x3_train = torch.cat([trainX_mfcc,trainX_chroma],dim=1)
# x3_test = torch.cat([testX_mfcc,testX_chroma],dim=1)

x4_train = trainX_db
x4_test = testX_db
# x3_train = trainX_chroma
# x3_test = testX_chroma

class myModel(nn.Module):
    def __init__(self):
        super(myModel, self).__init__()

        self.RNN = rnn(x1_train.size()[1],128, 1, 2)
        # self.RNN2 = rnn(x4_train.size()[1],128, 1, 2)
        self.CNN1 = cnn(1,5,6)
        self.CNN2 = cnn(1,2,2)
        self.CNN3 = cnn(1,2,1)
        self.CNN4 = cnn(1,2,1)


        # self.l1 = nn.Linear(7,1)
        # self.l5 = nn.Linear(13,1)
        # self.l6 = nn.Linear(53,1)

        self.lc = nn.Linear(8,1)
        # self.Linear_TempoBeat = linear()
    def forward(self,x1,x2,x4):

        # out1 = self.CNN1(x4)

        # out5 = self.l5(self.CNN2(x5))
        # out6 = self.l6(self.CNN3(x6))
        # out7 = self.CNN4(x7)
        # out = out1
        out2 = torch.squeeze(self.RNN(x1),dim=2)
        out = out2
        #
        # out = torch.squeeze(self.RNN2(x4)
        # out1 = self.CNN1(x3)
        # out = torch.squeeze(self.RNN(x1),dim=2)
        # out = torch.cat([out1, out2],dim=1)
        # a = self.RNN2(torch.unsqueeze(x2,dim=1))
        # out = self.l1(out)
        # out = self.lc(out)
        # out = F.softmax(out,dim=0)
        return out.double()


class cnn(nn.Module):
    def __init__(self, channel,k_size,stride):
        super(cnn, self).__init__()
        # Convolution 1 , input_shape=(input,28,28)
        self.cnn1 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=k_size, stride=stride)  # output_shape=(output,24,24)
        self.relu1 = nn.ReLU()  # activation
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)  # output_shape=(16,12,12)
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=k_size//2, stride=stride)  # output_shape=(32,8,8)
        self.relu2 = nn.ReLU()  # activation
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)  # output_shape=(32,4,4)
        # Fully connected 1 ,#input_shape=(32*4*4)

        # self.fc1 = nn.Linear(7,1)

    def forward(self, x):
        out = torch.unsqueeze(x,dim=1)
        # Convolution 1
        out = self.cnn1(out)
        out = self.relu1(out)
        # Max pool 1
        out = self.maxpool1(out)
        # Convolution 2
        out = self.cnn2(out)
        out = self.relu2(out)
        # Max pool 2
        out = self.maxpool2(out)
        out = out.view(x.size()[0], -1)
        # Linear function (readout)
        # out = self.fc1(out)
        return out

# class cnn(nn.Module):
#     def __init__(self,channel,kernel_size):
#         super(cnn, self).__init__()
#         self.cnn1 = nn.Conv2d(in_channels=channel,out_channels=channel,kernel_size=kernel_size)
#         self.avgpool1 = nn.AvgPool2d(kernel_size=2)
#
#         self.cnn2 = nn.Conv2d(in_channels=channel,out_channels=channel,kernel_size=kernel_size)
#         self.avgpool2 = nn.AvgPool2d(kernel_size=3)
#
#         # model.add(Dropout(rate=0.5))  # Add fully connected layer.
#         # model.add(Dense(64))
#         # model.add(Activation('relu'))
#         # model.add(Dropout(rate=0.5))  # Output layer
#         # model.add(Dense(10))
#         self.l1 = nn.Linear(105,64)
#         self.l2 = nn.Linear(64,1)
#     def forward(self,x):
#         x = torch.unsqueeze(torch.transpose(x,1,2),dim=1)
#         out = self.avgpool1(self.cnn1(x))
#         out = torch.mean(self.cnn2(out),dim=3)
#         out = torch.squeeze(out)
#         out = self.l2(self.l1(out))
#         return out

class linear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(linear, self).__init__()
        self.l1 = nn.Linear(input_dim,10)
        self.l2 = nn.Linear(10, 4)
        self.l3 = nn.Linear(4, output_dim)
    def forward(self, x):
        out = self.l3(self.l2(self.l1(x)))
        return out.double()

class rnn(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layer):
        super(rnn, self).__init__()
        # RNN Layer
        self.rnn = nn.RNN(input_dim, hidden_dim, n_layer)
        # self.rnn = nn.GRU(input_dim, hidden_dim, n_layer)
        # self.rnn = nn.LSTM(input_dim, hidden_dim, n_layer)
        # Fully connected layer
        self.l1 = nn.Linear(hidden_dim, hidden_dim//2)
        self.l2 = nn.Linear(hidden_dim//2, hidden_dim//4)
        self.l3 = nn.Linear(hidden_dim//4, output_dim)

    def forward(self, x):
        x = torch.unsqueeze(x,dim=1)
        out, _ = self.rnn(x, None)
        out = self.l1(out)
        out = self.l2(out)
        out = self.l3(out)
        # x = self.l3(self.l2(self.l1(x)))
        return out


if __name__ == '__main__':
    # train_x = torch.cat([trainX_views,trainX_zeroCrossRate, trainX_centroid], dim=1)
    # test_x = torch.cat([testX_views, testX_zeroCrossRate, testX_centroid], dim=1)

    model = myModel()

    # 2. loss and optimizer
    learning_rate = 0.0001
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    # 3. training loop
    epochs = 350
    for epoch in range(epochs):
        # forward pass and loss
        model.train()
        pred = model(x1_train,x2_train,x4_train)
        loss = criterion(pred,train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch%30==9:
            print(f'epoch: {epoch+1}, loss = {loss.item(): .4f}')

    model.eval()
    # prediction
    # y_pred2 = model(x1_test,x2_test,x3_test)

    y_pred2 = model(x1_test,x2_test,x4_test).detach().numpy()
    # print(y_pred2)
    # y_pred2 = np.expand_dims(y_pred2, axis=1)
    y_pred = y_pred2
    # y_pred = np.sum([0.1*y_pred2, 0.9*y_pred1], axis=0)

    # test_y = pd.DataFrame(y_pred)
    # test_y.columns = ['score']
    # ans = pd.DataFrame()
    # a = test.drop(test.columns.difference(['track']), axis=1)
    # ans = pd.concat([a, test_y], axis=1)
    # ans.to_csv('./result/re.csv',index=False)


    y_pred = np.sum([0.4 * y_pred2, 0.6 * y_pred1], axis=0)

    test_y = pd.DataFrame(y_pred)
    test_y.columns = ['score']
    ans = pd.DataFrame()
    a = test.drop(test.columns.difference(['track']), axis=1)
    ans = pd.concat([a, test_y], axis=1)
    ans.to_csv('./result/re.csv', index=False)
    ans1.to_csv('./result/result.csv', index=False)

    if predict_list is not None:
        prediction_dict = ans1.set_index('track').T.to_dict('list')
        for PATH in predict_list:
            _, tail = os.path.split(PATH)
            score = prediction_dict[tail][0]
            print('predict score of track {} = {}'.format(tail, score))
print()

print('done')