code structure

about .py files:
1.audio_process.py: 
process audio data using librosa and extract features in the clips from data/audios/clips/...
2.data_preprocess.py: 
process train/test dataset from data/..., connecting each data and their connected audio feature into pd.dataframe 
3.model.py: 
read features processed before and do steps including training model and prediction for test set
the prediction result will be saved in ./result/result.csv

how to run the code:
1. load audio data in audio_process.py and run the code 
2. load train/test.csv in data_preprocess.py and run the code
3. run model.py and get the prediction result

argument parsing:
cd to the hw file and input command below
python model.py --filename ...\audio1 ...\audio2 ...\audio3

where audio1, 2, 3,... are the clips in the test set that you want to predict the memorial scores
using blankspace to separate path of audio1, 2, 3,...
for example: 
python model.py --filename ...path1\normalize_5s_intro_0EVVKs6DQLo.wav ...path2\normalize_5s_intro_rWznOAwxM1g.wav ...path3\normalize_5s_intro_F64yFFnZfkI.wav 
and the result(score) will be printed on the screen 

