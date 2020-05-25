import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import lightgbm as lgb
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn import svm
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import Perceptron
from matplotlib import pyplot
import seaborn as sb
from sklearn import preprocessing
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support
#import plotly.plotly
#import plotly.figure_factory as ff



train = pd.read_csv("train.csv")
#test = pd.read_csv("test.csv")
weather = pd.read_csv("weather.csv")
spray = pd.read_csv("spray.csv")

train = train.drop_duplicates()
#test = test.drop_duplicates()
weather = weather.drop_duplicates()
spray = spray.drop_duplicates()

"""
analysis = weather[['Tmax','Tmin','Tavg','Depart','DewPoint','WetBulb','Heat','Cool','Sunrise','Sunset','SnowFall',
                    'PrecipTotal','StnPressure','SeaLevel','ResultSpeed','ResultDir','AvgSpeed','Station']]
plot=sb.pairplot(analysis,hue='Station')
plot.savefig("output.png")
fig = ff.create_scatterplotmatrix(analysis, diag='box', index='Station',
                                  height=800, width=800)
plotly.offline.iplot(fig, filename='Box plots along Diagonal Subplots')
"""

def create_month(x):
    return x.split('-')[1]
def create_day(x):
    return x.split('-')[2]
def create_year(x):
    return x.split('-')[0]


train['month'] = train.Date.apply(create_month)
train['day'] = train.Date.apply(create_day)

train['year'] = train.Date.apply(create_year)

weather['month'] = weather.Date.apply(create_month)
weather['day'] = weather.Date.apply(create_day)
weather['year'] = weather.Date.apply(create_year)

spray['month'] = spray.Date.apply(create_month)
spray['day'] = spray.Date.apply(create_day)
spray['year'] = spray.Date.apply(create_year)
spray=spray.drop('Time',axis=1)

weather_stn1 = weather[weather['Station']==1]
weather_stn2 = weather[weather['Station']==2]
weather_stn1 = weather_stn1.drop('Station', axis=1)
weather_stn2 = weather_stn2.drop('Station', axis=1)
weather = weather_stn1.merge(weather_stn2, on='Date')

weather = weather.replace('M', -1)
weather = weather.replace('-', -1)
weather = weather.replace('T', -1)
weather = weather.replace(' T', -1)
weather = weather.replace('  T', -1)

train['Lat_int'] = train.Latitude.apply(float)
train['Long_int'] = train.Longitude.apply(float)

spray['Lat_int'] = spray.Latitude.apply(float)
spray['Long_int'] = spray.Longitude.apply(float)

spray = spray.drop([ 'Longitude', 'Latitude'], axis = 1)
train = train.drop(['Address', 'AddressNumberAndStreet','Longitude', 'Latitude'], axis = 1)


train = train.merge(weather, on='Date')


train=train.drop('Date',axis=1)
weather=weather.drop('Date',axis=1)
spray=spray.drop('Date',axis=1)

#####################################################################

days = spray.day.unique()
monthes = spray.month.unique()
years = spray.year.unique()
lats = spray.Lat_int.unique()
longs = spray.Long_int.unique()

train['isSpray']=pd.Series([0])
train.isSpray=train['isSpray'].fillna(0)


d=np.array(spray.day)
m=np.array(spray.month)
y=np.array(spray.year)
lo=np.array(spray.Long_int)
la=np.array(spray.Lat_int)

st=np.zeros([len(d),5])
for i in range(len(d)):
    st[i][0] = d[i]
    st[i][1] = m[i]
    st[i][2] = y[i]
    st[i][3] = lo[i]
    st[i][4] = la[i]

s = st.astype(str)
st=[]
for i in range (len(s[:])):
    st.append(s[i][0]+s[i][1]+s[i][2]+s[i][3]+s[i][4])

spray['all'] = pd.Series(st)

##################################################################

d=np.array(train.day)
m=np.array(train.month)
y=np.array(train.year)
lo=np.array(train.Long_int)
la=np.array(train.Lat_int)

st=np.zeros([len(d),5])
for i in range(len(d)):
    st[i][0] = d[i]
    st[i][1] = m[i]
    st[i][2] = y[i]
    st[i][3] = lo[i]
    st[i][4] = la[i]

s = st.astype(str)
st=[]
for i in range (len(s[:])):
    st.append(s[i][0]+s[i][1]+s[i][2]+s[i][3]+s[i][4])

train['all'] = pd.Series(st)

for index,row in train.iterrows():
    #if spray['all'].isin([row['all']]).any():
    if row['all'] in set(spray['all']):
        train['isSpray'][index]=1

train = train.drop('all', axis=1)


########################################################################################################################

lbl = preprocessing.LabelEncoder()
lbl.fit(list(train['Species'].values))# + list(test['Species'].values))
train['Species'] = lbl.transform(train['Species'].values)


lbl.fit(list(train['Street'].values) )#+ list(test['Street'].values))
train['Street'] = lbl.transform(train['Street'].values)


lbl.fit(list(train['Trap'].values) )#+ list(test['Trap'].values))
train['Trap'] = lbl.transform(train['Trap'].values)



lbl.fit(list(train['CodeSum_x'].values))# + list(test['CodeSum_x'].values))
train['CodeSum_x'] = lbl.transform(train['CodeSum_x'].values)

lbl.fit(list(train['CodeSum_y'].values))# + list(test['CodeSum_y'].values))
train['CodeSum_y'] = lbl.transform(train['CodeSum_y'].values)



########################################################################################################################
train=train.astype(float)

#train = train.loc[:,(train != -1).any(axis=0)]

label=train.WnvPresent
train=train.drop('WnvPresent',axis=1)
sfm = SelectFromModel(LinearSVC(penalty='l1', loss='squared_hinge', dual=False))
data = sfm.fit_transform(train, label)
data = preprocessing.scale(data)
#data = preprocessing.scale(train)
transformer = FunctionTransformer(np.log1p, validate=True)
transformer.transform(data)
data = preprocessing.normalize(data, norm='l2')

feature_cols=train.columns
databackup=data
data = pd.DataFrame(sfm.inverse_transform(data),index=train.index, columns=feature_cols)
selCols = data.columns[data.var() !=0]
data = data[selCols]

TrainX, TestX, TrainY, TestY = train_test_split(data, label, test_size=0.2, random_state=1)
########################################################################################################################


def plotCurves(model):
    results = model.evals_result()
    epochs = len(results['validation_0']['auc'])
    x_axis = range(0, epochs)
    # plot auc
    fig, ax = pyplot.subplots()
    ax.plot(x_axis, results['validation_1']['auc'], label='Test')
    ax.legend()
    pyplot.ylabel('AUC')
    pyplot.title(' AUC by Epoch')
    pyplot.show()


########################################################################################################################

def train_lgb(Xtrain, Ytrain, Xvalid, Yvalid):
    dtrain = lgb.Dataset(Xtrain, label=Ytrain)
    dvalid = lgb.Dataset(Xvalid, label=Yvalid)
    param = {'num_leaves': 350, 'objective': 'binary', 'metric': 'auc'}
    print("Training model!")
    bst = lgb.train(param, dtrain, num_boost_round=1000, valid_sets=[dvalid],early_stopping_rounds=10, verbose_eval=False)
    valid_pred = bst.predict(Xvalid)
    valid_score = metrics.roc_auc_score(Yvalid, valid_pred)
    valid_score2 = metrics.f1_score(Yvalid, valid_pred.round(), average='weighted')
    valid_score3 = precision_recall_fscore_support(Yvalid, valid_pred.round(), average='weighted')
    print(f"Validation precision,recall,f score: ")
    print(valid_score3)
    print(f"Validation f1 score: {valid_score2:.4f}")
    print(f"Validation AUC score: {valid_score:.4f}")
    accuracy = accuracy_score(Yvalid, valid_pred.round())
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    return bst


bstm = train_lgb(TrainX,TrainY, TestX,  TestY)
#Validation f1 score: 0.9422
#Validation precision,recall,f score: (0.936804173868268, 0.9525528623001547, 0.9422460765080354)
#Validation AUC score: 0.8544
#Accuracy: 95.26%


def trainSVM(VTrainX,VTrainY, VTestX,  VTestY):
    clf = svm.SVC()
    clf.fit(VTrainX, VTrainY)
    valid_pred = clf.predict(VTestX)
    valid_score = metrics.roc_auc_score(VTestY, valid_pred)
    valid_score2 = metrics.f1_score(VTestY, valid_pred, average='weighted')
    valid_score3 = precision_recall_fscore_support(VTestY, valid_pred, average='weighted')
    print(f"Validation2 precision,recall,f score: ")
    print(valid_score3)
    print(f"Validation2 f1 score: {valid_score2:.4f}")
    print(f"Validation2 AUC score: {valid_score:.4f}")
    accuracy = accuracy_score(VTestY, valid_pred)
    print("Accuracy2: %.2f%%" % (accuracy * 100.0))
    return clf


svmm = trainSVM(TrainX,TrainY, TestX,  TestY)
#Validation2 f1 score: 0.9355
#Validation2 precision,recall,f score: (0.9152341357244328, 0.9566787003610109, 0.9354976221242358)
#Validation2 AUC score: 0.5000
#Accuracy2: 95.67%


def trainMLPClassifier(VTrainX,VTrainY, VTestX,  VTestY):
    clf = MLPClassifier()
    clf.fit(VTrainX, VTrainY)
    valid_pred = clf.predict(VTestX)
    valid_score = metrics.roc_auc_score(VTestY, valid_pred)
    valid_score2 = metrics.f1_score(VTestY, valid_pred, average='weighted')
    valid_score3 = precision_recall_fscore_support(VTestY, valid_pred, average='weighted')
    print(f"Validation3 precision,recall,f score: ")
    print(valid_score3)
    print(f"Validation3 f1 score: {valid_score2:.4f}")
    print(f"Validation3 AUC score: {valid_score:.4f}")
    accuracy = accuracy_score(VTestY, valid_pred)
    print("Accuracy3: %.2f%%" % (accuracy * 100.0))
    return clf


mlpm = trainMLPClassifier(TrainX,TrainY, TestX,  TestY)
#Validation3 f1 score: 0.9370
#Validation3 precision,recall,f score: (0.9262378524097238, 0.9494584837545126, 0.9358213608182486)
#Validation3 AUC score: 0.5257
#Accuracy3: 95.15%


def trainRFC(VTrainX,VTrainY, VTestX,  VTestY):
    clf = RandomForestClassifier()
    clf.fit(VTrainX, VTrainY)
    valid_pred = clf.predict(VTestX)
    valid_score = metrics.roc_auc_score(VTestY, valid_pred)
    valid_score2 = metrics.f1_score(VTestY, valid_pred, average='weighted')
    valid_score3 = precision_recall_fscore_support(VTestY, valid_pred, average='weighted')
    print(f"Validation4 precision,recall,f score: ")
    print(valid_score3)
    print(f"Validation4 f1 score: {valid_score2:.4f}")
    print(f"Validation4 AUC score: {valid_score:.4f}")
    accuracy = accuracy_score(VTestY, valid_pred)
    print("Accuracy4: %.2f%%" % (accuracy * 100.0))
    return clf


RFC = trainRFC(TrainX,TrainY, TestX,  TestY)
#Validation4 f1 score: 0.9351
#Validation4 precision,recall,f score: (0.9226221026504545, 0.9422382671480144, 0.9317024661667764)
#Validation4 AUC score: 0.5393
#Accuracy4: 94.48%


def trainXGBClassifier(VTrainX,VTrainY, VTestX,  VTestY):
    clf = XGBClassifier()
    eval_set = [(VTrainX, VTrainY), (VTestX, VTestY)]
    clf.fit(VTrainX, VTrainY, eval_metric="auc", eval_set=eval_set, verbose=True)
    valid_pred = clf.predict(VTestX)
    valid_score = metrics.roc_auc_score(VTestY, valid_pred)
    valid_score2 = metrics.f1_score(VTestY, valid_pred,average='weighted')
    valid_score3 = precision_recall_fscore_support(VTestY, valid_pred, average='weighted')
    print(f"Validation5 precision,recall,f score: ")
    print(valid_score3)
    print(f"Validation5 AUC score: {valid_score:.4f}")
    print(f"Validation5 f1 score: {valid_score2:.4f}")
    accuracy = accuracy_score(VTestY, valid_pred)
    print("Accuracy5: %.2f%%" % (accuracy * 100.0))
    fig, ax = pyplot.subplots(figsize=(10, 15))
    xgb.plot_importance(clf, ax=ax)
    return clf


XGB = trainXGBClassifier(TrainX,TrainY, TestX,  TestY)
plotCurves(XGB)
#Validation5 AUC score: 0.5747
#Validation5 precision,recall,f score: (0.934487052273894, 0.9473955647240846, 0.939973895547217)
#Validation5 f1 score: 0.9400
#Accuracy5: 94.74%


def trainKNeighborsClassifier(VTrainX,VTrainY, VTestX,  VTestY):
    clf = KNeighborsClassifier()
    clf.fit(VTrainX, VTrainY)
    valid_pred = clf.predict(VTestX)
    valid_score = metrics.roc_auc_score(VTestY, valid_pred)
    valid_score2 = metrics.f1_score(VTestY, valid_pred, average='weighted')
    valid_score3 = precision_recall_fscore_support(VTestY, valid_pred, average='weighted')
    print(f"Validation6 precision,recall,f score: ")
    print(valid_score3)
    print(f"Validation6 f1 score: {valid_score2:.4f}")
    print(f"Validation6 AUC score: {valid_score:.4f}")
    accuracy = accuracy_score(VTestY, valid_pred)
    print("Accuracy6: %.2f%%" % (accuracy * 100.0))
    return clf


KNC = trainKNeighborsClassifier(TrainX,TrainY, TestX,  TestY)
#Validation6 f1 score: 0.9346
#Validation6 precision,recall,f score: (0.9248659197086221, 0.9473955647240846, 0.9346355669480494)
#Validation6 AUC score: 0.5236
#Accuracy6: 94.74%



def RadialBasisFunctionKernel(VTrainX,VTrainY, VTestX,  VTestY):
    kernel = 1.0 * RBF(1.0)
    clf =GaussianProcessClassifier(kernel=kernel,random_state=0).fit(VTrainX, VTrainY)
    valid_pred = clf.predict(VTestX)
    valid_score = metrics.roc_auc_score(VTestY, valid_pred)
    valid_score2 = metrics.f1_score(VTestY, valid_pred, average='weighted')
    valid_score3 = precision_recall_fscore_support(VTestY, valid_pred, average='weighted')
    print(f"Validation7 precision,recall,f score: ")
    print(valid_score3)
    print(f"Validation7 f1 score: {valid_score2:.4f}")
    print(f"Validation7 AUC score: {valid_score:.4f}")
    accuracy = accuracy_score(VTestY, valid_pred)
    print("Accuracy7: %.2f%%" % (accuracy * 100.0))
    return clf

RBC = RadialBasisFunctionKernel(TrainX,TrainY, TestX,  TestY)



def gaussianNaiveBayesClassifier(VTrainX,VTrainY, VTestX,  VTestY):
    clf =GaussianNB().fit(VTrainX, VTrainY)
    valid_pred = clf.predict(VTestX)
    valid_score = metrics.roc_auc_score(VTestY, valid_pred)
    valid_score2 = metrics.f1_score(VTestY, valid_pred, average='weighted')
    valid_score3 = precision_recall_fscore_support(VTestY, valid_pred, average='weighted')
    print(f"Validation8 precision,recall,f score: ")
    print(valid_score3)
    print(f"Validation8 f1 score: {valid_score2:.4f}")
    print(f"Validation8 AUC score: {valid_score:.4f}")
    accuracy = accuracy_score(VTestY, valid_pred)
    print("Accuracy8: %.2f%%" % (accuracy * 100.0))
    return clf

GNBC = gaussianNaiveBayesClassifier(TrainX,TrainY, TestX,  TestY)
#Validation8 precision,recall,f score: (0.9463349163744859, 0.6477565755544095, 0.7503711104334898, None)
#Validation8 f1 score: 0.7504
#Validation8 AUC score: 0.7136
#Accuracy8: 64.78%


def LinearDiscriminantClassifier(VTrainX,VTrainY, VTestX,  VTestY):
    clf =LinearDiscriminantAnalysis().fit(VTrainX, VTrainY)
    valid_pred = clf.predict(VTestX)
    valid_score = metrics.roc_auc_score(VTestY, valid_pred)
    valid_score2 = metrics.f1_score(VTestY, valid_pred, average='weighted')
    valid_score3 = precision_recall_fscore_support(VTestY, valid_pred, average='weighted')
    print(f"Validation9 precision,recall,f score: ")
    print(valid_score3)
    print(f"Validation9 f1 score: {valid_score2:.4f}")
    print(f"Validation9 AUC score: {valid_score:.4f}")
    accuracy = accuracy_score(VTestY, valid_pred)
    print("Accuracy9: %.2f%%" % (accuracy * 100.0))
    return clf

LDC = LinearDiscriminantClassifier(TrainX,TrainY, TestX,  TestY)
#Validation9 precision,recall,f score: (0.9310833952199952, 0.9443011861784425, 0.9369735331733785, None)
#Validation9 f1 score: 0.9370
#Validation9 AUC score: 0.5617
#Accuracy9: 94.43%



def PerceptronClassifier(VTrainX,VTrainY, VTestX,  VTestY):
    clf =Perceptron(tol=1e-3, random_state=0).fit(VTrainX, VTrainY)
    valid_pred = clf.predict(VTestX)
    valid_score = metrics.roc_auc_score(VTestY, valid_pred)
    valid_score2 = metrics.f1_score(VTestY, valid_pred, average='weighted')
    valid_score3 = precision_recall_fscore_support(VTestY, valid_pred, average='weighted')
    print(f"Validation10 precision,recall,f score: ")
    print(valid_score3)
    print(f"Validation10 f1 score: {valid_score2:.4f}")
    print(f"Validation10 AUC score: {valid_score:.4f}")
    accuracy = accuracy_score(VTestY, valid_pred)
    print("Accuracy10: %.2f%%" % (accuracy * 100.0))
    return clf

PC = PerceptronClassifier(TrainX,TrainY, TestX,  TestY)
#Validation10 precision,recall,f score: (0.9150839742583582, 0.9530685920577617, 0.9336901179124099, None)
#Validation10 f1 score: 0.9337
#Validation10 AUC score: 0.4981
#Accuracy10: 95.31%


























#test['month'] = test.Date.apply(create_month)
#test['day'] = test.Date.apply(create_day)
#test['year'] = train.Date.apply(create_year)
#test['Lat_int'] = test.Latitude.apply(float)
#test['Long_int'] = test.Longitude.apply(float)
#test = test.drop([ 'Address', 'AddressNumberAndStreet','Longitude', 'Latitude'], axis = 1)
#test = test.merge(weather, on='Date')
#test=test.drop('Date',axis=1)
"""
test['isSpray']=pd.Series([0])
test.isSpray=test['isSpray'].fillna(0)

d=np.array(test.day)
m=np.array(test.month)
y=np.array(test.year)
lo=np.array(test.Long_int)
la=np.array(test.Lat_int)

st=np.zeros([len(d),5])
for i in range(len(d)):
    st[i][0] = d[i]
    st[i][1] = m[i]
    st[i][2] = y[i]
    st[i][3] = lo[i]
    st[i][4] = la[i]

s = st.astype(str)
st=[]
for i in range (len(s[:])):
    st.append(s[i][0]+s[i][1]+s[i][2]+s[i][3]+s[i][4])

test['all'] = pd.Series(st)

for index,row in test.iterrows():
    if row['all'] in set(spray['all']):
        test['isSpray'][index]=1

test = test.drop('all', axis=1)
"""
#test['Species'] = lbl.transform(test['Species'].values)
#test['Street'] = lbl.transform(test['Street'].values)
#test['Trap'] = lbl.transform(test['Trap'].values)
#test['CodeSum_x'] = lbl.transform(test['CodeSum_x'].values)
#test['CodeSum_y'] = lbl.transform(test['CodeSum_y'].values)
#test = test.loc[:,(test != -1).any(axis=0)]
