import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Get url from DVC
import dvc.api

# path = 'dvc_data/AdSmartABdata.csv'
# #FIXME: We will have to change the path  according to the path of our local machines
# repo = 'C:/Users/Kamuzinzi N. Egide/Documents/Ten_academy/week 2/Ad-campaign-performance'
# version = 'v1'

# data_url = dvc.api.get_url(
#     path = path,
#     repo = repo,
#     rev = version
# )
mlflow.set_experiment('Logistic Regression model ')




def feature_data(df):
    
    browser_feature_df = df[["experiment", "hour", "date", 'device_make', 'browser', 'yes']] 
    platform_feature_df = df[["experiment", "hour", "date", 'device_make', 'platform_os', 'yes']] 

    return browser_feature_df, platform_feature_df

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)


    # data = pd.read_csv(data_url, sep=";")
    # print(data.columns)
    data = pd.read_csv("data/encoded_data.csv")   
    data = data[data.columns.tolist()[2:]]
    print(data.columns)


    # cleaned_data = encode_labels(df)

    X = data.iloc[:,:-1] # Features
    y = data.yes # Target variable
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=16)





    # instantiate the model (using the default parameters)
    logreg = LogisticRegression(random_state=16)


    # TODO: This will be changed to our model
    with mlflow.start_run():

            # fit the model with data
        logreg.fit(train_x, train_y)

        predicted_qualities = logreg.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)


        # print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)
        print("  Accuracy: %s" % metrics.accuracy_score(test_y,predicted_qualities))

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("Accuracy",metrics.accuracy_score(test_y,predicted_qualities)) 

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(logreg, "model", registered_model_name="ElasticnetWineModel")
        else:
            mlflow.sklearn.log_model(logreg, "model")