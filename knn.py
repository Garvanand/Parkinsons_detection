import pandas
import sklearn
from sklearn.model_selection import cross_validate,cross_val_score,train_test_split
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

url = "data.csv"
features = ["MDVP:Fo(Hz)","MDVP:Fhi(Hz)","MDVP:Flo(Hz)","MDVP:Jitter(%)","MDVP:Jitter(Abs)","MDVP:RAP","MDVP:PPQ","Jitter:DDP","MDVP:Shimmer","MDVP:Shimmer(dB)","Shimmer:APQ3","Shimmer:APQ5","MDVP:APQ","Shimmer:DDA","NHR","HNR","RPDE","DFA","spread1","spread2","D2","PPE","status"]
dataset = pandas.read_csv(url, names = features)

array = dataset.values
scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(array)
X = scaled[:,0:22]
Y = scaled[:,22]
validation_size = 0.25
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size = validation_size, random_state = seed)
print(X_train)

num_folds = 10
num_instances = len(X_train)
seed = 7
scoring = 'accuracy'

results = []
clf = KNeighborsClassifier()
kfold = sklearn.model_selection.KFold(n_splits=num_instances,random_state = seed)
cv_results = cross_val_score(clf, X_train, Y_train, cv = kfold, scoring = scoring)
clf.fit(X_train, Y_train)
predictions = clf.predict(X_validation)
print("KNN")
print(accuracy_score(Y_validation, predictions)*100)
print(matthews_corrcoef(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
