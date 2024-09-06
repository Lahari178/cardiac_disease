import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,cohen_kappa_score,matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_curve,precision_recall_curve
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_curve
df = pd.read_csv('C:\\Users\\LAHARI\\OneDrive\\Desktop\\Dataset.csv')
#reading dataset using pandas
#Df
#showing details
df.info()
#to find count of null values
df.isnull().sum()
#size of data
df.shape
#count of male and female
df.sex.value_counts()
#count of diseased and non-diseased
df.target.value_counts()
pd.crosstab(df.target,df.sex)
#standard measures for every row
df.describe().T
#Box Plot for Finding Outliers
col1=[df['restecg'],df['oldpeak'],df['slope'],df['ca']]
fig,ax=plt.subplots()
ax.boxplot(col1,patch_artist=True,vert=False)
plt.yticks([1,2,3,4],['restecg','oldpeak','slope','ca'])
plt.show()
col2=[df['thalach'],df['chol'],df['trestbps'],df['age']]
fig,ax=plt.subplots()
ax.boxplot(col2,patch_artist=True,vert=False)
plt.yticks([1,2,3,4],['thalach','chol','trestbps','age'])
plt.show()
def removeOutliers(data, col):
    q3=data[col].quantile(0.75)
    q1=data[col].quantile(0.25)
    IQR=q3-q1
    Lowr=q1-(1.5*IQR)
    upr=q3+(1.5*IQR)
    return upr,Lowr
u1, l1=removeOutliers(df,"chol")
print(u1,l1)
df=df[(df['chol']>l1)&(df['chol']<u1)]
print("after change:",df.shape)
u2,l2=removeOutliers(df,"trestbps")
print(u2,l2)
df=df[(df['trestbps']>l2)&(df["trestbps"]<u2)]
print(df.shape)
u3,l3=removeOutliers(df,'thalach')
print(u3,l3)
df=df[(df['thalach']>l3)&(df['thalach']<u3)]
print(df.shape)
u4,l4=removeOutliers(df,'ca')
print(u4,l4)
df=df[(df['ca']>l4)&(df['ca']<u4)]
print(df.shape)
u5,l5=removeOutliers(df,'oldpeak')
print(u5,l5)
df=df[(df['oldpeak']>l5)&(df['oldpeak']<u5)]
print(df.shape)
col2=[df['thalach'],df['chol'],df['trestbps'],df['age']]
fig,ax=plt.subplots()
ax.boxplot(col2,patch_artist=True,vert=False)
plt.yticks([1,2,3,4],['thalach','chol','trestbps','age'])
plt.show()
col1=[df['restecg'],df['oldpeak'],df['slope'],df['ca']]
fig,ax=plt.subplots()
ax.boxplot(col1,patch_artist=True,vert=False)
plt.yticks([1,2,3,4],['restecg','oldpeak','slope','ca'])
plt.show()
df.shape
x=df.drop("target",axis=1).values
y=df["target"].values
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
X_train=StandardScaler().fit_transform(X_train)
X_test=StandardScaler().fit_transform(X_test)
print("Number of transactions in X_train dataset: ", X_train.shape)
print("Number of transactions in y_train dataset: ", y_train.shape)
print("Number of transactions in X_test dataset: ", X_test.shape)
print("Number of transactions in y_test dataset: ", y_test.shape)
sam1 = LogisticRegression()
sam1.fit(X_train, y_train)
sam1pred = sam1.predict(X_test)
print(classification_report(y_test, sam1pred))
bos=sum(y_train==1)
print("Before OverSampling, counts of label '1':", bos)
aos= sum(y_train == 0)
print("Before OverSampling, counts of label '0':",aos)
smt = SMOTE(random_state = 2)
X_train_res, y_train_res = smt.fit_resample(X_train, y_train)
print("\nAfter OverSampling, counts of label '1': ",bos)
print("After OverSampling, counts of label '0': ",aos)
logre1 = LogisticRegression()
logre1.fit(X_train_res, y_train_res)
sample = logre1.predict(X_test)
print(classification_report(y_test, sample))
sns.heatmap(df.corr(),annot=True,annot_kws={'size':10},linewidth=0,linecolor="white")
sns.set(rc={"figure.figsize":(10,5)})
fig1=sns.FacetGrid(df,hue="target")
fig1.map(sns.kdeplot,'age',fill=True)
fig1.set(xlim=(20,80))
plt.legend(labels=['Diseased','Non-Diseased'])
fig1.set(ylabel=('Density'))
warnings.filterwarnings("ignore")
logr = LogisticRegression()
logr.fit(X_train, y_train)
importance = logr.coef_[0]
x = ['restecg', 'sex', 'cp', 'trestbps', 'chol', 'thalach', 'exang', 'oldpeak', 'age', 'fbs', 'slope', 'ca', 'thal']
for i, v in enumerate(importance):
    print("Feature: %s, Score: %.5f" % (x[i], v))
plt.bar(x, importance)
plt.show()

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
imp2 = dt.feature_importances_
x = ['thalach', 'exang', 'trestbps', 'chol', 'fbs', 'restecg', 'oldpeak', 'age', 'sex', 'cp', 'slope', 'ca', 'thal']
for a, b in enumerate(imp2):
    print("Feature: %s, Score: %.5f" % (x[a], b))

plt.bar(x, imp2)
plt.show()

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
imp3 = rf.feature_importances_
x = ['slope', 'ca', 'age', 'trestbps', 'chol', 'fbs', 'sex', 'cp', 'restecg', 'thalach', 'exang', 'oldpeak', 'thal']
for x_idx, y in enumerate(imp3):
    print("Feature: %s, Score: %.5f" % (x[x_idx], y))

plt.bar(x, imp3)
plt.show()
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)
imp4 = classifier.feature_importances_
for p, q in enumerate(imp4):
    print("Feature: %s, Score: %.5f" % (x[p], q))

plt.bar(x, imp4)
plt.show()