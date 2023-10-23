import csv
import torch
import pandas as pd
import numpy as np


# read feature
feature = torch.load('./audio_feature/feature.pth')
audio_dict = feature['audio_dict']
# audio = feature[audio]
f1_stft = feature['f1_stft']
f2_db = feature['f2_db']
f3_zeroCrossRate = feature['f3_zeroCrossRate']
f4_centroid = feature['f4_centroid']
f5_rolloff = feature['f5_rolloff']
f6_mfcc = feature['f6_mfcc']
f7_bandwidth = feature['f7_bandwidth']
f8_contrast = feature['f8_contrast']
f9_flat = feature['f9_flat']
f10_tempo = feature['f10_tempo']
f11_beat = feature['f11_beat']
f12_chroma = feature['f12_chroma']


# 對應train資料 (220 songs)
file = open('./data/train.csv')
csvreader = csv.reader(file)
# [track, views, country, score]
train_header = next(csvreader)
# train feature and train label
train_x, train_y = [], []
for row in csvreader:
    # track id
    audio_id = audio_dict[row[0]]
    # train feature
    train_x.append([row[0], audio_id, int(row[1]), row[2], f1_stft[audio_id].view(np.float32), f2_db[audio_id],
                    f3_zeroCrossRate[audio_id], f4_centroid[audio_id], f5_rolloff[audio_id],
                    f6_mfcc[audio_id], f7_bandwidth[audio_id], f8_contrast[audio_id],
                    f9_flat[audio_id], f10_tempo[audio_id], f11_beat[audio_id],f12_chroma[audio_id]
                    ])
    # train label (score)
    train_y.append(float(row[3]))

# 轉成dataframe (train_x為歌曲與對應特徵，train_y為歌曲對應的score)
train_x = pd.DataFrame(train_x)
train_y = pd.DataFrame(train_y)
train_x.columns = ['track', 'song_id', 'views', 'country', 'stft', 'db', 'zeroCrossRate',
                   'centroid', 'rolloff', 'mfcc', 'bandwidth', 'contrast', 'flat', 'tempo', 'beat','chroma']
train_y.columns = ['score']
# 將country轉為dummies
country_dummies = pd.get_dummies(train_x['country'], prefix='Coun', prefix_sep='_')
train_x = train_x.drop('country', axis=1)
train_x = pd.concat([train_x,country_dummies], axis=1)
# 查看train_x, train_y資訊
print(train_x.info())
print(train_y.info())
del row, audio_id, country_dummies, file, csvreader, train_header



# 對應test資料 (15 songs)
file = open('./data/test.csv')
csvreader = csv.reader(file)
# [track, views, country]
test_header = next(csvreader)
# test feature (no label)
test_x = []
for row in csvreader:
    # track id
    audio_id = audio_dict[row[0]]
    # test feature (song_id:Int, view:Int, country:String, tempo:Float, beat/centroid/mfcc:Tensor)
    test_x.append([row[0], audio_id, int(row[1]), row[2], f1_stft[audio_id].view(np.float32), f2_db[audio_id],
                    f3_zeroCrossRate[audio_id], f4_centroid[audio_id], f5_rolloff[audio_id],
                    f6_mfcc[audio_id], f7_bandwidth[audio_id], f8_contrast[audio_id],
                    f9_flat[audio_id], f10_tempo[audio_id], f11_beat[audio_id],f12_chroma[audio_id]
                    ])

# 轉成dataframe (train_x為歌曲與對應特徵，train_y為歌曲對應的score)
test_x = pd.DataFrame(test_x)
test_x.columns = ['track', 'song_id', 'views', 'country', 'stft', 'db', 'zeroCrossRate',
                   'centroid', 'rolloff', 'mfcc', 'bandwidth', 'contrast', 'flat', 'tempo', 'beat', 'chroma']

country_dummies = pd.get_dummies(test_x['country'], prefix='Coun', prefix_sep='_')
test_x = test_x.drop('country', axis=1)
test_x = pd.concat([test_x,country_dummies], axis=1)
# print(test_x.info())

# train的country dummies比test多，補上沒有的country
miss_country = set(train_x.columns) - set(test_x.columns )
# Add a missing column in test set with default value equal to 0
for c in miss_country:
    test_x[c] = 0
# Ensure the order of column in the test set is in the same order than in train set
test_x = test_x[train_x.columns]
print(test_x.info())



train_x = train_x.drop('song_id', axis=1)
test_x = test_x.drop('song_id', axis=1)
train_x.to_csv('./processed_data/train/train_x.csv')
train_y.to_csv('./processed_data/train/train_y.csv')
test_x.to_csv('./processed_data/test/test_x.csv')




# train_x的各種feature存成tensor
country_list = ['Coun_AE', 'Coun_AU','Coun_BO', 'Coun_BR', 'Coun_CA', 'Coun_CH', 'Coun_CL', 'Coun_CO',
       'Coun_DE', 'Coun_EG', 'Coun_FR', 'Coun_GB', 'Coun_GE', 'Coun_GH', 'Coun_GR', 'Coun_HK', 'Coun_HR', 'Coun_HU',
        'Coun_ID', 'Coun_IN', 'Coun_IT', 'Coun_JP', 'Coun_KR', 'Coun_LB', 'Coun_MA', 'Coun_MD',
       'Coun_MX', 'Coun_N/A', 'Coun_NL', 'Coun_PH', 'Coun_PL', 'Coun_PT',
       'Coun_RO', 'Coun_TH', 'Coun_TN', 'Coun_TR', 'Coun_UG', 'Coun_US',
       'Coun_VN', 'Coun_ZA']
country = torch.tensor(train_x[country_list].values).float()
views = torch.unsqueeze(torch.tensor(train_x['views']),dim=1).float()
stft = torch.tensor(train_x['stft']).float()
db = torch.tensor(train_x['db']).float()
zeroCrossRate  = torch.tensor(train_x['zeroCrossRate'])
zeroCrossRate = torch.unsqueeze(torch.sum(zeroCrossRate, dim=1), dim=1).float()
centroid = torch.tensor(train_x['centroid'])
centroid = torch.squeeze(centroid).float()
rolloff = torch.tensor(train_x['rolloff'])
rolloff = torch.squeeze(rolloff).float()
mfcc = torch.tensor(train_x['mfcc']).float()
bandwidth = torch.tensor(train_x['bandwidth'])
bandwidth = torch.squeeze(bandwidth).float()
contrast = torch.tensor(train_x['contrast']).float()
flat = torch.tensor(train_x['flat'])
flat = torch.squeeze(flat).float()
tempo = torch.unsqueeze(torch.tensor(train_x['tempo']), dim=1).float()
beat = torch.tensor(train_x['beat']).float()
chroma = torch.tensor(train_x['chroma']).float()





# mfcc = torch.reshape(mfcc, (mfcc.size()[0],mfcc.size()[1]*mfcc.size()[2]))
store = {'audio_dict':audio_dict, 'country':country, 'views':views,
         'f1_stft':stft, 'f2_db':db, 'f3_zeroCrossRate':zeroCrossRate,
         'f4_centroid':centroid, 'f5_rolloff':rolloff, 'f6_mfcc':mfcc,
         'f7_bandwidth':bandwidth, 'f8_contrast':contrast, 'f9_flat':flat,
         'f10_tempo':tempo, 'f11_beat':beat, 'f12_chroma':chroma
        }
torch.save(store, "./processed_data/train/train_x.pth")

# test_x的各種feature存成tensor
country = torch.tensor(test_x[country_list].values).float()
views = torch.unsqueeze(torch.tensor(test_x['views']),dim=1).float()
stft = torch.tensor(test_x['stft']).float()
db = torch.tensor(test_x['db']).float()
zeroCrossRate  = torch.tensor(test_x['zeroCrossRate'])
zeroCrossRate = torch.unsqueeze(torch.sum(zeroCrossRate, dim=1), dim=1).float()
centroid = torch.tensor(test_x['centroid'])
centroid = torch.squeeze(centroid).float()
rolloff = torch.tensor(test_x['rolloff'])
rolloff = torch.squeeze(rolloff).float()
mfcc = torch.tensor(test_x['mfcc']).float()
bandwidth = torch.tensor(test_x['bandwidth'])
bandwidth = torch.squeeze(bandwidth).float()
contrast = torch.tensor(test_x['contrast']).float()
flat = torch.tensor(test_x['flat'])
flat = torch.squeeze(flat).float()
tempo = torch.unsqueeze(torch.tensor(test_x['tempo']), dim=1).float()
beat = torch.tensor(test_x['beat']).float()
chroma = torch.tensor(test_x['chroma']).float()

store = {'country':country, 'views':views,
         'f1_stft':stft, 'f2_db':db, 'f3_zeroCrossRate':zeroCrossRate,
         'f4_centroid':centroid, 'f5_rolloff':rolloff, 'f6_mfcc':mfcc,
         'f7_bandwidth':bandwidth, 'f8_contrast':contrast, 'f9_flat':flat,
         'f10_tempo':tempo, 'f11_beat':beat, 'f12_chroma':chroma
        }
torch.save(store, "./processed_data/test/test_x.pth")
print('done')
