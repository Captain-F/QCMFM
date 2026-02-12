from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import re
import pickle
import numpy as np
from sklearn.preprocessing import LabelBinarizer

def read_pickle(path):
    with open(path, 'rb')as f:
        feats = pickle.load(f)
        #feats = joblib.load(f)
    return feats

def save_pickle(path, feats):
    with open(path, 'wb')as f:
        pickle.dump(feats, f)

def sort_p(paths):
    paths_ = sorted(paths, key=lambda x: int(re.split(r'[\\.]', str(x))[-2]))
    return paths_



def read_txt(path):

    with open(r"stopwords.txt", "r", encoding="utf8") as f:
        stopwords = f.read().split()

    with open(path, "r", encoding="utf8") as f:
        txt = f.read()
    txt_ = "".join([t for t in txt if t not in stopwords])

    return txt_

def read_stopwords(path):
    with open(path, "r", encoding="utf8") as f:
        txt = f.read()


falang_img_p = list(Path(r"img_path").glob('*.png'))
jinyin_img_p = list(Path(r"img_path").glob('*.png'))
qiqi_img_p = list(Path(r"img_path").glob('*.png'))
tongqi_img_p = list(Path(r"img_path").glob('*.png'))
yushi_img_p = list(Path(r"img_path").glob('*.png'))

falang_img_arr = [cv2.imread(str(p)) for p in falang_img_p]
falang_img_res = [cv2.resize(img_arr, (224, 224)) for img_arr in falang_img_arr]
falang_img = np.array(falang_img_res).astype(np.float32)/ 255.0

jinyin_img_arr = [cv2.imread(str(p)) for p in jinyin_img_p]
jinyin_img_res = [cv2.resize(img_arr, (224, 224)) for img_arr in jinyin_img_arr]
jinyin_img = np.array(jinyin_img_res).astype(np.float32)/ 255.0

qiqi_img_arr = [cv2.imread(str(p)) for p in qiqi_img_p]
qiqi_img_res = [cv2.resize(img_arr, (224, 224)) for img_arr in qiqi_img_arr]
qiqi_img = np.array(qiqi_img_res).astype(np.float32)/ 255.0

tongqi_img_arr = [cv2.imread(str(p)) for p in tongqi_img_p]
tongqi_img_res = [cv2.resize(img_arr, (224, 224)) for img_arr in tongqi_img_arr]
tongqi_img = np.array(tongqi_img_res).astype(np.float32)/ 255.0

yushi_img_arr = [cv2.imread(str(p)) for p in yushi_img_p]
yushi_img_res = [cv2.resize(img_arr, (224, 224)) for img_arr in yushi_img_arr]
yushi_img = np.array(yushi_img_res).astype(np.float32)/ 255.0

# 获取所有图片的描述；

falang_txt_p = list(Path(r"txt_path").glob('*.txt'))
jinyin_txt_p = list(Path(r"txt_path").glob('*.txt'))
qiqi_txt_p = list(Path(r"txt_path").glob('*.txt'))
tongqi_txt_p = list(Path(r"txt_path"").glob('*.txt'))
yushi_txt_p = list(Path(r"txt_path").glob('*.txt'))

falang_txt = [read_txt(p) for p in falang_txt_p]
jinyin_txt = [read_txt(p) for p in jinyin_txt_p]
qiqi_txt = [read_txt(p) for p in qiqi_txt_p]
tongqi_txt = [read_txt(p) for p in tongqi_txt_p]
yushi_txt = [read_txt(p) for p in yushi_txt_p]


# 获取描述文本特征；

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Concatenate, GlobalAveragePooling2D, Flatten, Dense, LSTM, Conv2D, ResNet50
import numpy as np
import tensorflow_hub as hub
import tensorflow_text as text

txt_inputs = Input(shape=(), dtype=tf.string)
preprocessor = hub.KerasLayer(r'bert_path')
encoder_inputs = preprocessor(txt_inputs)
encoder = hub.KerasLayer(r'bert_path',
                         trainable=False)
encoder_outputs = encoder(encoder_inputs)
#txt_outputs = encoder_outputs['pooled_output']
txt_outputs = encoder_outputs["sequence_output"]

model = Model(inputs=txt_inputs, outputs=txt_outputs)

falang_bert = model.predict(falang_txt, batch_size=8)
jinyin_bert = model.predict(jinyin_txt, batch_size=8)
qiqi_bert = model.predict(qiqi_txt, batch_size=8)
tongqi_bert = model.predict(tongqi_txt, batch_size=8)
yushi_bert = model.predict(yushi_txt, batch_size=8)