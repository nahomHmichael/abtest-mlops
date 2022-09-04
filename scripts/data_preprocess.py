import pandas as pd
import numpy as np


class Data_preprocess:
   
    def __init__(self, df: pd.DataFrame) -> None:
        """
        Returns a data_preProcess Object with the passed DataFrame Data set as its own DataFrame
        """
        self.df = df

    def remove_duplicates(self) -> pd.DataFrame:
        """
        Returns a DataFrame where duplicate rows are removed
        """
        removables = self.df[self.df.duplicated()].index
        return self.df.drop(index=removables, inplace=True)

    def fix_outlier(self, column: str) -> pd.DataFrame:
        """
        Returns a DataFrame where outlier of the specified column is fixed
        """
        self.df[column] = np.where(self.df[column] > self.df[column].quantile(
            0.95), self.df[column].median(), self.df[column])

        return self.df

    def fix_outlier_columns(self, columns: list) -> pd.DataFrame:
        """
        Returns a DataFrame where outlier of the specified columns is fixed
        """
        try:
            for column in columns:
                self.df[column] = np.where(self.df[column] > self.df[column].quantile(
                    0.95), self.df[column].median(), self.df[column])
        except:
            print("Cant fix outliers for each column")

        return self.df

    def remove_unwanted_columns(self, columns: list) -> pd.DataFrame:
        """
        Returns a DataFrame where the specified columns in the list are removed
        """
        self.df.drop(columns, axis=1, inplace=True)
        return self.df

    def change_columns_type_to(self, cols: list, data_type: str) -> pd.DataFrame:
        """
        Returns a DataFrame where the specified columns data types are changed to the specified data type
        """
        try:
            for col in cols:
                self.df[col] = self.df[col].astype(data_type)
        except:
            print('Failed to change columns type')

        return self.df

    def save_clean_data(self, name: str):
        """
        The objects dataframe gets saved with the specified name 
        """
        try:
            self.df.to_csv(name)

        except:
            print("Failed to save data")