import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from urllib.parse import urlparse
import dagshub
dagshub.init("mlflow-repo", "Phuocphenikaa", mlflow=True)
mlflow.start_run()

class opt:
    file_data = r"C:\Users\phuoc\Downloads\flow\data\train.csv"
    columns_drop = ["Name","Ticket","Cabin"]
    drop_nan = True
    column_label = "Embarked"
    valid_spit = 0.2

class test:
        file_data = r"C:\Users\phuoc\Downloads\flow\data\test.csv"
        columns_drop = ["Name", "Ticket", "Cabin"]
        drop_nan = True
        column_label = None
def clean_df(opt):
    df = pd.read_csv(opt.file_data).drop(columns = opt.columns_drop)
    if opt.drop_nan:
        df = df.dropna()
    for col in df.columns:
        if df.dtypes[col] == object:
          value_list = df.loc[:,col].unique()
          for i,value in enumerate(value_list):
              df.loc[df[col]==value,col] = np.float64(i)
        df[col] = df[col].astype(np.float64)
    if opt.column_label:
        df[opt.column_label] = df[opt.column_label].astype(np.int64)
    x = df.iloc[:, :-1].values
    x = (x-np.min(x,axis=0))/(x.max(axis = 0)-x.min(axis = 0))
    y = df.iloc[:, -1].values
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=opt.valid_spit,shuffle=True)
    return x_train,x_test,y_train,y_test

def accuracy(y_predict,y):
    assert (y_predict.shape==y.shape)
    return np.sum(np.where(y==y_predict,1,0))/y.shape[0]

x_train,x_valid,y_train,y_valid = clean_df(opt)

n_estimators = 100
model = RandomForestClassifier(n_estimators=n_estimators)
model = model.fit(x_train,y_train)
y_predict = model.predict(x_valid)
accuracy = accuracy(y_predict,y_valid)

signature = infer_signature(x_train, y_train)
mlflow.log_metric("accuracy",accuracy)
mlflow.log_param("n",n_estimators)
mlflow.sklearn.log_model(
        model, "model", registered_model_name="phuoc-model", signature=signature
)


mlflow.end_run()