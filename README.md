# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Detect File Encoding: Use chardet to determine the dataset's encoding.
2. Load Data: Read the dataset with pandas.read_csv using the detected encoding.
3. Inspect Data: Check dataset structure with .info() and missing values with .isnull().sum().
4. Split Data: Extract text (x) and labels (y) and split into training and test sets using train_test_split.
5. Convert Text to Numerical Data: Use CountVectorizer to transform text into a sparse matrix.
6. Train SVM Model: Fit an SVC model on the training data.
7. Predict Labels: Predict test labels using the trained SVM model.
8.Evaluate Model: Calculate and display accuracy with metrics.accuracy_score.
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: THAMEEZ AHAMED A
RegisterNumber: 212224220116
*/

import chardet
file = 'spam.csv'
with open(file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result
import pandas as pd
data = pd.read_csv("spam.csv",encoding= 'Windows-1252')
data.head()
data.isnull().sum()
data.info()
X=data["v2"].values
y=data["v1"].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
X_train
X_test
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)
X_train
X_test
from sklearn.svm import SVC
svc=SVC()
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print("Accuracy :",accuracy)
from sklearn import metrics
confusion = metrics.confusion_matrix(y_test,y_pred)
print("Confusion Matrix :\n",confusion)
from sklearn import metrics
cl = metrics.classification_report(y_test,y_pred)
print("Classification Report :\n",cl)
```

## Output:
<img width="1405" height="52" alt="image" src="https://github.com/user-attachments/assets/60557315-de0e-4e25-ab3d-835e6574a360" />
<img width="1407" height="262" alt="image" src="https://github.com/user-attachments/assets/bf6b3767-14f5-40fc-b862-f6a8a503c080" />
<img width="1424" height="167" alt="image" src="https://github.com/user-attachments/assets/795a0038-3382-4823-8757-9e668e45856a" />
<img width="1392" height="316" alt="image" src="https://github.com/user-attachments/assets/371ae42b-5ff3-4cbd-a2fc-08eafbf18ae3" />
<img width="1398" height="244" alt="image" src="https://github.com/user-attachments/assets/a8c7cc93-73ba-4fb5-9f2e-32c7f50f027d" />
<img width="1400" height="295" alt="image" src="https://github.com/user-attachments/assets/08c0eec5-8407-4d66-b318-de9c1aa69963" />
<img width="1390" height="80" alt="image" src="https://github.com/user-attachments/assets/a89ee98b-dbbf-4fcd-bfb6-6130d899c864" />
<img width="1426" height="77" alt="image" src="https://github.com/user-attachments/assets/25ee8301-1915-4283-a78e-f02938a05093" />
<img width="1387" height="60" alt="image" src="https://github.com/user-attachments/assets/8a4e9868-6675-4ac1-a704-60eba7d61e79" />
<img width="1407" height="57" alt="image" src="https://github.com/user-attachments/assets/024ef865-3131-40ad-8dfe-08922af4bc7e" />
<img width="1397" height="97" alt="image" src="https://github.com/user-attachments/assets/2350b659-5e6c-45be-b8e9-015d2d122ef6" />
<img width="1384" height="265" alt="image" src="https://github.com/user-attachments/assets/1317db3f-7986-45ad-b7ec-21e910f10ec5" />



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
