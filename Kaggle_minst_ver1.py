import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.model_selection import  train_test_split

print(os.listdir("../input"))


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")

X = train_df.drop(['label'],axis=1)
y = train_df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=9)
y_train = to_categorical(y_train, 10)
X_train = X_train / 255.0
X_test = X_test / 255.0

def Model(x,y):
    model=Sequential([
        Dense(units=64,input_shape=(784,),activation='relu')
        Dense(units=10,activation='softmax')
    ])
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(x ,y, batch_size=1890, epochs=20)
    return model
    
model = Model(X_train,Y_train)
class_predict = model.predict(X_test)

ans=[]
for i in range(len(class_predict)):
    ans.append(np.argmax(class_predict[i]))

print(accuracy_score(Y_test, ans))

pX = test / 255.0

predict = model.predict(pX)

anser = []
for i in range(len(predict)):
    anser.append(np.argmax(predict[i]))
    
submit = pd.read_csv("../input/sample_submission.csv")
submit['Label'] = anser
submit.to_csv('submission.csv', index=False)
