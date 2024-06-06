# pipeline_utilities
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#from sklearn.metrics import mean_squared_error, r2_score
#from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
#from sklearn.linear_model import LinearRegression
#import statsmodels.api as sm
import glob

def load_data():
    # Get a list of all CSV files in a directory
    csv_files = glob.glob('data/*.csv')

    combined_df = pd.DataFrame()

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        combined_df = pd.concat([combined_df, df])
        first_column_value_counts = combined_df.iloc[:,0].value_counts()
        first_column_shape = combined_df.shape
        print("Loading the DataFrame")
        display(combined_df)
        display('value counts'.title(),first_column_value_counts)
        display('shape'.title(),first_column_shape)
    return combined_df


def X_feature_set():
    data = load_data()
    X = data.copy().dropna().drop(columns=data.columns[0])
    print("Display X below:")
    display(X)
    y = data[data.columns[0]].values.reshape(-1,1)
    print("y")
    display(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    print("X_train")
    display(X_train)
    print("X_test")
    display(X_test)
    print("y_train")
    display(y_train)
    print("y_test")
    display(y_test)
    return train_test_split(X, y)

def StandardScaler():
    X_scaler = StandardScaler().fit(X_train)
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
    print("X_train_scaled")
    display(X_train_scaled)
    print("X_test_scaled")
    display(X_test_scaled)
    return X_train_scaled, X_test_scaled













