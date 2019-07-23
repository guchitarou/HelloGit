# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import  train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score

train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
Y=train['label']
X=train.drop(['label'],axis=1)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,random_state=9)
Y_train = to_categorical(Y_train,10)
X_train=X_train/255
X_test=X_test/255


def Model(x,y):
    model=Sequential()
    model.add(Dense(units=64,input_shape=(784,),activation='relu'))
    model.add(Dense(units=10,activation='softmax'))
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(x,y,batch_size=1890,epochs=20)
    return model
    

model=Model(X_train,Y_train)
class_predict=model.predict(X_test)

ans=[]
for i in range(len(class_predict)):
    ans.append(np.argmax(class_predict[i]))

print(accuracy_score(Y_test, ans))

pX=test/255

predict=model.predict(pX)

anser=[]
for i in range(len(predict)):
    anser.append(np.argmax(predict[i]))
    
submit = pd.read_csv("../input/sample_submission.csv")
submit['Label'] =anser
submit.to_csv('submission.csv', index=False)
