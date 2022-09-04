
import pandas as pd
import numpy as np


class Df_description:
    def __init__(self, df: pd.DataFrame):
        """
            Returns a dataframe Info Object with the passed DataFrame Data
            Parameters
        """
        self.df = df

    def find_matrix_correlation(self):
        '''
            Returns the correlation matrix of the passed Dataframe
        '''
        return self.df.corr()

    def find_memory_usage(self):
        '''
            Returns the memory usage of the passed DAtaframe
        '''
        print(f"Current DataFrame Memory Usage of columns is :")
        return self.df.memory_usage()

    def find_total_memory_usage(self):
        '''
            Returns the total memory usage of the passed Dataframe
        '''
        value = self.df.memory_usage(deep=True).sum()
        print(f"Current DataFrame Memory Usage:\n{value}")
        return value

    def find_aggregate(self, stat_list: list):
        '''
            Returns the aggregate of the passed Dataframe
        '''
        try:
            return self.df.agg(stat_list)
        except:
            print("Failed to get aggregates")

    def find_dataframe_columns_unique_value_count(self):
        '''
            Returns the unique value count of the passed Dataframe
        '''
        return pd.DataFrame(self.df.apply(lambda x: len(x.value_counts(dropna=False)), axis=0), columns=['Unique Value Count']).sort_values(by='Unique Value Count', ascending=True)

    def find_duplicates(self):
        '''
            Returns the duplicates of the passed Dataframe
        '''
        return self.df[self.df.duplicated()]

    def find_column_based_missing_percentage(self):
        '''
            Returns the missing percentage of the passed Dataframe
        '''
        col_null = self.df.isnull().sum()
        total_entries = self.df.shape[0]
        missing_percentage = []
        for col_missing_entries in col_null:
            value = str(
                round(((col_missing_entries/total_entries) * 100), 2)) + " %"
            missing_percentage.append(value)

        missing_df = pd.DataFrame(col_null, columns=['total_missing_values'])
        missing_df['missing_percentage'] = missing_percentage
        return missing_df

    def find_columns_missing_percentage_greater_than(self, num: float):
        '''
            Returns the missing percentage of the passed Dataframe
        '''
        all_cols = self.get_column_based_missing_percentage()
        extracted = all_cols['missing_percentage'].str.extract(r'(.+)%')
        return extracted[extracted[0].apply(lambda x: float(x) >= num)].index

    def find_columns_with_missing_values(self):
        '''
            Returns the missing vlaue of the passed Dataframe
        '''
        lst = self.df.isnull().any()
        arr = []
        index = 0
        for col in lst:
            if col == True:
                arr.append(lst.index[index])
            index += 1
        return arr

    def find_column_based_missing_values(self):
        '''
            Returns the missing value of the passed dataframe
        '''
        value = self.df.isnull().sum()
        df = pd.DataFrame(value, columns=['missing_count'])
        df.drop(df[df['missing_count'] == 0].index, inplace=True)
        df['type'] = [self.df.dtypes.loc[i] for i in df.index]
        return df