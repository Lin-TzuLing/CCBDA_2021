import librosa
import numpy as np
import os
import torch



# load songs as vectors (235*110250)
audio = []
audio_dict = {}
directory = './data/audios/clips/'
# given audio file id
i = 0
for filename in os.scandir(directory):
    if filename.is_file():
        # 235 songs in the file
        y, _ = librosa.load(path=filename,mono=True)
        audio_dict[filename.name] = i
        audio.append(y)
        i += 1
audio =  np.array(audio)
del y, directory, filename, i

# get feature of songs (f1~f11)
f1_stft, f2_db, f3_zeroCrossRate = [], [], []
f4_centroid, f5_rolloff, f6_mfcc = [], [], []
f7_bandwidth, f8_contrast, f9_flat = [], [], []
f10_tempo, f11_beat, f12_chroma = [], [], []
for i in range(len(audio)):
    y = audio[i]
    # 短時傅里葉變換
    stft = librosa.stft(y=y)
    # 色度頻率
    chroma_stft = librosa.feature.chroma_stft(y=y)
    # 分貝
    db = librosa.amplitude_to_db(abs(stft))
    # 過零率(金屬、搖滾等高衝擊聲音會有更高的值)
    zeroCrossRate = librosa.zero_crossings(y=y, pad=False)
    # 頻譜中心(值越小，表明越多的頻譜能量集中在低頻範圍內)
    centroid = librosa.feature.spectral_centroid(y=y)
    # 頻譜中心做normalization
    # centroid = sklearn.preprocessing.minmax_scale(centroid, axis=0)
    # 頻譜滾降點
    rolloff = librosa.feature.spectral_rolloff(y=y)
    # MFCC(梅爾頻率倒譜系數)
    mfcc = librosa.feature.mfcc(y=y)
    # 帶寬
    bandwidth = librosa.feature.spectral_bandwidth(y=y)
    # 譜對比度
    contrast = librosa.feature.spectral_contrast(y=y)
    # 譜平坦度
    flat = librosa.feature.spectral_flatness(y=y)

    tempo, beat = librosa.beat.beat_track(y=y)
    # pad beats到一樣長度
    beat = librosa.frames_to_time(beat)
    beat = np.pad(beat, (0,17-len(beat)), mode='constant',constant_values=0)

    # chroma_stft = librosa.feature.chroma_stft(y=y)
    # rmse = librosa.feature.rms(y=y)

    # 11 feature
    f1_stft.append(stft)
    f2_db.append(db)
    f3_zeroCrossRate.append(zeroCrossRate)
    f4_centroid.append(centroid)
    f5_rolloff.append(rolloff)
    f6_mfcc.append(mfcc)
    f7_bandwidth.append(bandwidth)
    f8_contrast.append(contrast)
    f9_flat.append(flat)
    f10_tempo.append(tempo)
    f11_beat.append(beat)
    f12_chroma.append(chroma_stft)

del i, tempo, beat, centroid, y, mfcc, stft, db, zeroCrossRate, rolloff, bandwidth, contrast, flat, chroma_stft





# write to pth
store = {'audio_dict':audio_dict,'audio': audio,
         'f1_stft':f1_stft, 'f2_db':f2_db, 'f3_zeroCrossRate':f3_zeroCrossRate,
         'f4_centroid':f4_centroid, 'f5_rolloff':f5_rolloff, 'f6_mfcc':f6_mfcc,
         'f7_bandwidth':f7_bandwidth, 'f8_contrast':f8_contrast, 'f9_flat':f9_flat,
         'f10_tempo':f10_tempo, 'f11_beat':f11_beat, 'f12_chroma':f12_chroma
        }
torch.save(store, "./audio_feature/feature.pth")
print('done')

