import pandas as pd
import numpy as np


class DataPreprocess:

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def remove_duplicates(self) -> pd.DataFrame:
        remv = self.df[self.df.duplicated()].index
        return self.df.drop(index=remv, inplace=True)

    def fix_outlier(self, column: str) -> pd.DataFrame:
        self.df[column] = np.where(self.df[column] > self.df[column].quantile(
            0.95), self.df[column].median(), self.df[column])

        return self.df

    def fix_outlier_columns(self, columns: list) -> pd.DataFrame:
       # Returns a DataFrame where outlier of the specified columns is fixed
        try:
            for column in columns:
                self.df[column] = np.where(self.df[column] > self.df[column].quantile(
                    0.95), self.df[column].median(), self.df[column])
        except:
            print("Error fixing outliers for columns")

        return self.df

    def remove_unwanted_columns(self, columns: list) -> pd.DataFrame:
        # Returns a DataFrame where the specified columns in the list are removed
       
        self.df.drop(columns, axis=1, inplace=True)
        return self.df

    def change_columns_type_to(self, cols: list, data_type: str) -> pd.DataFrame:
        # Returns a DataFrame where the specified columns data types are changed to the specified data type
        try:
            for col in cols:
                self.df[col] = self.df[col].astype(data_type)
        except:
            print('Error changing columns type')

        return self.df

    def save_clean_data(self, name: str):
        # The objects dataframe gets saved with the specified name 
        try:
            self.df.to_csv(name)

        except:
            print("Error saving data!")
