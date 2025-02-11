import numpy as np
import pandas as pd  

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import classification_report

def run_model_training(X_train, X_test, y_train, y_test):
    # Train/fit model on the training set
    reg = LinearRegression(fit_intercept=True) # True means y-intercept will be by the line of best fit
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    #print(classification_report(y_test, y_pred))
    print("--------------------------------------")
    print("Model performance on 'test' set")
    print("--------------------------------------")
    print("- Test set score: {:.2f}".format(reg.score(X_test, y_test)))
    print("\n")
    return reg

def prepare_data(path_to_data, encoding="latin-1"):
    """
        
    """
    # Read data from path
    data = pd.read_csv(path_to_data, encoding=encoding)
    X = data['RM']
    y = data['target']
    return {'RM':X, 
            'target':y}
    

    
def create_train_test_data(X, y, test_size, random_state):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=test_size, 
                                                        random_state=random_state)

    return {'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test}
