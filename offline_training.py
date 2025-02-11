import numpy as np
import pandas as pd  
from sklearn.metrics import mean_squared_error, r2_score

import joblib
from joblib import load
from model import train_builder


# custom package
if __name__ == "__main__":
    # 0.Path to data
    path_to_data = './data/boston_dataset.csv'

    # 1.Prepare the data
    prepared_data = train_builder.prepare_data(path_to_data, encoding="latin-1")

    # 2.Create train - test split
    train_test_data = train_builder.create_train_test_data(prepared_data['RM'], 
                                            prepared_data['target'], 
                                            0.3, 2021)
    X_train=pd.DataFrame(train_test_data['X_train'])
    X_test=pd.DataFrame(train_test_data['X_test'])
    y_train=pd.DataFrame(train_test_data['y_train'])
    y_test=pd.DataFrame(train_test_data['y_test'])
    # 3.Run training
    model = train_builder.run_model_training(X_train=X_train, X_test=X_test, 
                            y_train=y_train, y_test=y_test)
    ## model = train_builder.run_model_training(train_test_data['X_train'], train_test_data['X_test'], 
    ##                        train_test_data['y_train'], train_test_data['y_test'])

    # 4.Save the trained model and vectorizer
    joblib.dump(model, './model/regressor.pkl')
    print("Finished training model")
