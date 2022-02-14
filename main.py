import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import boston_housing
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas
import matplotlib.pyplot as plt

# 1978년 미국 보스턴 지역의 주택 가격이며, 506개 지역의 주택 가격 중앙값을 1,000달러 단위로 나타냈습니다.
# 범죄율, 주택당 방의 수, 고속도로 접근성, 학생/교사 비율 등 13가지 데이터를 이용합니다.
(X_train, Y_train), (x_test, y_test) = boston_housing.load_data()
# 트레이닝 세트의 데이터 개수는 404, 테스트 데이터셋의 개수는 102개임을 알수있음.
len(X_train)
len(x_test)
#  X의 데이터셋의 경우는 위에서 나열한 13개의 컬럼값을 담고있고
#  Y의 데이터셋은 1000달러단위 집값을 표기함

## 13개의 정보, 가격의 단위의 범위가 너무 넓기때문에 정규화가 필요함
# 정규화는 평균값을 빼준뒤 표준편차를 나눠주면 됨.
X_mean = X_train.mean()
X_std = X_train.std()
X_train -= X_mean
X_train /= X_std
x_test -= X_mean
x_test /= X_std

Y_mean = Y_train.mean()
Y_std = Y_train.std()
Y_train -= Y_mean
Y_train /= Y_std
y_test -= Y_mean
y_test /= Y_std
print(X_train[0], Y_train[0], x_test[0], y_test[0])

# 순차모델 생성
# 13개의 컬럼을 모두삽입
model = keras.Sequential([
    keras.layers.Dense(units=50, activation='relu', input_shape=(13,)),
    keras.layers.Dense(units=30, activation='relu'),
    keras.layers.Dense(units=20, activation='relu'),
    keras.layers.Dense(units=1)])

# 모델 컴파일
# adam = 가중치와 절편을 추정하여 실제값과 유사한값을 찾음
model.compile(optimizer=keras.optimizers.Adam(lr=0.07), loss='mse')
model.summary()

# 모델 학습시키기
m1_fit = model.fit(X_train, Y_train, epochs=25, batch_size=32, validation_split=0.25)
# 머신러닝 성능 결과보기
model.evaluate(x_test, y_test)
# 예측값 보기
plt.plot(m1_fit.history['loss'], 'b-', label='loss')
plt.plot(m1_fit.history['val_loss'], 'r--', label='val_loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()
## 실제 예측값 보기
pred = model.predict(x_test)

plt.figure(figsize=(5,5))
plt.plot(y_test, pred, 'b.')
plt.axis([min(y_test), max(y_test), min(y_test), max(y_test)])

# y=x에 해당하는 대각선
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], ls="--", c=".3")
plt.xlabel('test_Y')
plt.ylabel('pred_Y')
plt.show()
# 이상치들도 많이 발생이되고 예측이 잘 되지않음을 알수있음

# 머신러닝 수정하기
model = keras.Sequential([
    keras.layers.Dense(units=52, activation='relu', input_shape=(13,)),
    keras.layers.Dense(units=39, activation='relu'),
    keras.layers.Dense(units=26, activation='relu'),
    keras.layers.Dense(units=1)])

model.compile(optimizer=keras.optimizers.Adam(lr=0.07), loss='mse')
# 콜백함수는 과적합 방지를 해줄수있는 메서드임.
m2_fit = model.fit(X_train, Y_train, epochs=25, batch_size=32, validation_split=0.25,
                    callbacks=[keras.callbacks.EarlyStopping(patience=3, monitor='val_loss')])

# 바뀐모델 확인
plt.plot(m2_fit.history['loss'], 'b-', label='loss')
plt.plot(m2_fit.history['val_loss'], 'r--', label='val_loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()
# 성능평가
model.evaluate(x_test,y_test)
# 실제 예측값과 비교학
pred_Y = model.predict(x_test)

plt.figure(figsize=(5,5))
plt.plot(y_test, pred_Y, 'b.')
plt.axis([min(y_test), max(y_test), min(y_test), max(y_test)])

plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], ls="--", c=".3")
plt.xlabel('test_Y')
plt.ylabel('pred_Y')
plt.show()