import pandas as pd
from dataclasses import dataclass

@dataclass
class CreditDatasetConfig:
    data_path: str = "data/credit_data.csv"
    filters: dict = None
    random_seed: int = 42
    test_size: float = 0.2
    lowest_age: int = 18
    highest_age: int = 100
    max_income_delta_percentage: float = 500.0
    min_income_delta_percentage: float = -100.0
    categorical_cols: list = ['marital_status', 'jobtitle_category', 'address_category', 'gender']
    numerical_cols: list = ['income_delta_percentage', 'age']
    outlier_cols: list = ['income_delta_percentage']

class CreditDataPreprocessor:
    def __init__(self, data, config: CreditDatasetConfig):
        self.data_original = data
        self.data_current = data.copy()
        self.documentation = {}
        self.config = config

    def __filter_categorical(self):
        """
        Filters out rows from the DataFrame based on specified categorical values.
        Args:
            filters (dict): A dictionary where keys are column names and values are lists of categorical values to filter out.
        example:
            filters = {
                'column1': ['value1', 'value2'],
                'column2': ['value3']
            }
        """
        if 'filtering_steps' not in self.documentation:
            self.documentation['filtering_steps'] = []
        for column, values_to_remove in self.config.filters.items():
            self.data_current = self.data_current[~self.data_current[column].isin(values_to_remove)]
            self.documentation['filtering_steps'].append(f"Filtered categorical values in column: {column} - Removed values: {values_to_remove}")
    def __remove_duplicates(self):
        """
        Removes duplicate rows from the DataFrame.
        """
        if 'filtering_steps' not in self.documentation:
            self.documentation['filtering_steps'] = []
        initial_shape = self.data_current.shape
        self.data_current = self.data_current.drop_duplicates().reset_index(drop=True)
        final_shape = self.data_current.shape
        self.documentation['filtering_steps'].append(f"Removed duplicates - Rows removed: {initial_shape[0] - final_shape[0]}")
    def __handle_invalid_ages(self):
        if 'filtering_steps' not in self.documentation:
            self.documentation['filtering_steps'] = []
        initial_shape = self.data_current.shape
        self.data_current = self.data_current[(self.data_current['age'] >= self.config.lowest_age) & (self.data_current['age'] <= self.config.highest_age)].reset_index(drop=True)
        final_shape = self.data_current.shape
        self.documentation['filtering_steps'].append(f"Removed invalid ages - Rows removed: {initial_shape[0] - final_shape[0]}")
    def __handle_extreme_income_delta(self):
        if 'filtering_steps' not in self.documentation:
            self.documentation['filtering_steps'] = []
        initial_shape = self.data_current.shape
        self.data_current = self.data_current[(self.data_current['income_delta_percentage'] >= self.config.min_income_delta_percentage) & (self.data_current['income_delta_percentage'] <= self.config.max_income_delta_percentage)].reset_index(drop=True)
        final_shape = self.data_current.shape
        self.documentation['filtering_steps'].append(f"Removed extreme income delta percentages - Rows removed: {initial_shape[0] - final_shape[0]}")
    def __handle_due_principal_vs_credit_limit(self):
        if 'filtering_steps' not in self.documentation:
            self.documentation['filtering_steps'] = []
        high_due_before = len(self.data_current[self.data_current['due_principal'] > self.data_current['credit_limit']])
        if high_due_before > 0:
            # Cap due_principal to credit_limit
            self.data_current.loc[self.data_current['due_principal'] > self.data_current['credit_limit'], 'due_principal'] = self.data_current['credit_limit']
            self.documentation['filtering_steps'].append(f"Due principal capped to credit limit for affected records")
    def __missing_categorical_values_imputation(self):
        if 'filtering_steps' not in self.documentation:
            self.documentation['filtering_steps'] = []
        for i, col in enumerate(self.config.categorical_cols):
            if col in self.data_current.columns:
                missing_count = self.data_current[col].isnull().sum()
                if missing_count > 0:
                    self.data_current[col] = self.data_current[col].fillna('Unknown')
                    self.documentation['filtering_steps'].append(f"\n5.{i}. Filled {missing_count} missing values in {col} with 'Unknown'...")
    def __missing_numerical_values_imputation(self):
        if 'filtering_steps' not in self.documentation:
            self.documentation['filtering_steps'] = []
        for i, col in enumerate(self.config.numerical_cols):
            if col in self.data_current.columns:
                missing_count = self.data_current[col].isnull().sum()
                if missing_count > 0:
                    median_val = self.data_current[col].median()
                    self.data_current[col] = self.data_current[col].fillna(median_val)
                    self.documentation['filtering_steps'].append(f"\n6.{i}. Filled {missing_count} missing values in {col} with median ({median_val:.2f})...")
    def __remove_outliers(self):
        if 'filtering_steps' not in self.documentation:
            self.documentation['filtering_steps'] = []
        for i, col in enumerate(self.config.outlier_cols):
            if col in self.data_current.columns:
                mean_val = self.data_current[col].mean()
                std_val = self.data_current[col].std()
                outlier_mask = (self.data_current[col] < mean_val - 3*std_val) | (self.data_current[col] > mean_val + 3*std_val)
                outliers_count = outlier_mask.sum()
                if outliers_count > 0:
                    initial_shape = self.data_current.shape
                    self.data_current = self.data_current[~outlier_mask].reset_index(drop=True)
                    final_shape = self.data_current.shape
                    self.documentation['filtering_steps'].append(f"\n7.{i}. Removed {outliers_count} outliers from {col} - Rows removed: {initial_shape[0] - final_shape[0]}")
    def __cleaning_summary(self):
        if 'cleaning_summary' not in self.documentation:
            self.documentation['cleaning_summary'] = {}
        self.documentation['cleaning_summary'].append(f"Final dataset shape: {self.data_current.shape}")
        self.documentation['cleaning_summary'].append(f"Records removed during cleaning: {self.data_original.shape[0] - self.data_current.shape[0]}")
        self.documentation['cleaning_summary'].append(f"Data quality improvement completed!")
    def preprocess(self):
        self.__filter_categorical()
        self.__remove_duplicates()
        self.__handle_invalid_ages()
        self.__handle_extreme_income_delta()
        self.__handle_due_principal_vs_credit_limit()
        self.__missing_categorical_values_imputation()
        self.__missing_numerical_values_imputation()
        self.__remove_outliers()
        self.__cleaning_summary()
        for key, value in self.documentation.items():
            if isinstance(value, list):
                print(f"\n{key.upper()}:")
                for item in value:
                    print(f"- {item}")
            else:
                print(f"\n{key.upper()}: {value}")
        return self.data_current