import os
import pandas as pd
import tensorflow as tf
from datetime import datetime

# GPU 충돌을 막기위한 Device 설정
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# Training, Test DataSet 불러오기
tr_set = pd.read_csv("training.csv")
test_set = pd.read_csv("test.csv")

print(tr_set.columns)
tr_set.head()

# Feature, Target, Test 설정하기
feature = tr_set[['temperature']]
target = tr_set[['outer_type']]
test = test_set[['temperature']]

# NN 을 위한 Layer 배정하기
X = tf.keras.layers.Input(shape=[1])
H = tf.keras.layers.Dense(10, activation='swish')(X) # 10계층의 HiddenLayer 사용
y = tf.keras.layers.Dense(1)(H)

# Model 정의 및 학습
model = tf.keras.models.Model(X,y)
model.compile(loss='mse')
model.fit(feature, target, epochs=100)

# Test DataSet 을 활용한 결과 검증
print(model.predict(test[:5]))

breakpoint