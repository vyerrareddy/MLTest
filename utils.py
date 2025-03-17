import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from numpy import log
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, r2_score ,mean_absolute_error
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# for feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.base import BaseEstimator, TransformerMixin

# import libraries for model validation
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from CONSTANTS import FILE_PATH , num_boost_round , xgb_params 

#**********************************************************************************

def read_convert_file(file_path:str)->pd.DataFrame:
  """
  Takes a CSV format file path and returns a DataFrame with DatetimeIndex 
  Args: file path

  Returns: DataFrame with DatetimeIndex 
  """
  df = pd.read_csv(FILE_PATH, parse_dates= True, dayfirst= False) 
  df['BusinessDate'] = pd.to_datetime(df['BusinessDate'])
  df = df.set_index('BusinessDate')
  return df 

#**********************************************************************************
def create_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    From DataTime column of the original DF create features for TIme Series

    Args:
        df (pd.DataFrame): Input DataFrame with a DatetimeIndex.

    Returns:
        pd.DataFrame: DataFrame with additional time series features.

    Raises:
        ValueError: If input DataFrame does not have a DatetimeIndex.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Input DataFrame must have a DatetimeIndex.")

    df = df.copy()

    features = {

        'month': df.index.month,
        'year': df.index.year,
        'dayofmonth': df.index.day

    }

    for feature_name, feature_values in features.items():
        df[feature_name] = feature_values

    return df
#**********************************************************************************
def converting_dtypes (df: pd.DataFrame )->pd.DataFrame:
    """
    This function converts all the object dtypes  into 
    numericals. 
    Args: Input is a Pandas Dataframe
    
    Returns : pd.DataFrame : DataFrame with converting object dtype to numbers 
    
    """
    for col in df.select_dtypes(include='object').columns:
        if df[col].nunique() <= 6:
            df[col] = pd.factorize(df[col])[0] 
    return df


#**********************************************************************************
def dropping_columns (df: pd.DataFrame )->pd.DataFrame:
    """
    This function removes certain columns, and resets the index. 
    Args: Input is a Pandas Dataframe
    
    Returns : pd.DataFrame : DataFrame after removing columns and resetting index
    
    """
    df.drop(['FiscalYearNumber', 'RunKey'], axis=1, inplace=True)
    df.reset_index (inplace = True,drop= True ) 
    return df

#**********************************************************************************

def trainTestSplit (df: pd.DataFrame, target_column: str = 'Variable') -> pd.DataFrame:
    """
    This Function splits the dataframe into  Training and testing split 
    of 80% for training and 20% for testing. This also splits the data into
    dependent and independent variables X ,y 
    
    Args: pd.DataFrame 
    """
    if target_column not in df.columns:
        raise ValueError (f"Target column '{target_column}' not found in dataframe")
    X = df.drop([target_column], axis=1)
    y = (df[target_column] )
    X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42) 

    return X_train, y_train, X_test, y_test
    
    
#**********************************************************************************

def calculate_metrics (data_tuple, model):
    """ Calculates and returns training and test metrics. """
    
    X_train, y_train, X_test, y_test = data_tuple 
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)  
    
    train_preds = model.predict(dtrain)
    test_preds = model.predict(dtest) 
    
    train_rmse = mean_squared_error(y_train, train_preds)
    test_rmse = mean_squared_error(y_test, test_preds) 
    
    train_r2 = r2_score(y_train, train_preds)
    test_r2 = r2_score(y_test, test_preds) 
    
    metrics = {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
    } 
     
    return metrics 




#**********************************************************************************
class XGBTrainer (BaseEstimator, TransformerMixin):
    """ Wrapper class for training an XGBoost model in a scikit-learn pipeline.  """
    
    def __init__ (self, xgb_params, num_boost_round =  999):
        self.xgb_params = xgb_params 
        self.num_boost_round = num_boost_round
        self.model = None 
    
    def fit (self, X_y ,y= None):
        
        """
        Trains an XGBoost model.

        Args:
            X_y (tuple): Tuple containing (X_train, y_train, X_test, y_test)

        Returns:
            self: Trained model
        """
        X_train, y_train, X_test, y_test  = X_y 
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test) 
        self.model =  xgb.train(
                self.xgb_params,
                dtrain,
                num_boost_round=self.num_boost_round,
                evals=[(dtest, "Test")],
                early_stopping_rounds=10) 
        return self 
    def transform(self, X_y):
        """ Returns the trained XGBoost model. """
        return self.model 
    def predict (self, X):
        if self.model is None:
            raise ValueError("Model is not trained. Call fit() first. ")
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest) 
    def get_model(self):
        return self.model
# -------------------------------------------------------------------------------

