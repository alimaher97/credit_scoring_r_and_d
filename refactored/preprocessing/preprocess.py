import pandas as pd
from dataclasses import dataclass

@dataclass
class CreditDatasetConfig:
    data_path: str = "data/credit_data.csv"
    random_seed: int = 42
    test_size: float = 0.2
    lowest_age: int = 18
    highest_age: int = 100
    max_income_delta_percentage: float = 500.0
    min_income_delta_percentage: float = -100.0
class CreditDataPreprocessor:
    def __init__(self, data, config: CreditDatasetConfig):
        self.data_original = data
        self.data_current = data.copy()
        self.documentation = {}
        self.lowest_age = config.lowest_age
        self.highest_age = config.highest_age
    def filter_categorical(self, filters):
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
        for column, values_to_remove in filters.items():
            self.data_current = self.data_current[~self.data_current[column].isin(values_to_remove)]
            self.documentation['filtering_steps'].append(f"Filtered categorical values in column: {column} - Removed values: {values_to_remove}")

    def __remove_duplicates(self):
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
        self.data_current = self.data_current[(self.data_current['age'] >= self.lowest_age) & (self.data_current['age'] <= self.highest_age)].reset_index(drop=True)
        final_shape = self.data_current.shape
        self.documentation['filtering_steps'].append(f"Removed invalid ages - Rows removed: {initial_shape[0] - final_shape[0]}")
    def __handle_extreme_income_delta(self, min_val, max_val):
        if 'filtering_steps' not in self.documentation:
            self.documentation['filtering_steps'] = []
        initial_shape = self.data_current.shape
        self.data_current = self.data_current[(self.data_current['income_delta_percentage'] >= min_val) & (self.data_current['income_delta_percentage'] <= max_val)].reset_index(drop=True)
        final_shape = self.data_current.shape
        self.documentation['filtering_steps'].append(f"Removed extreme income delta percentages - Rows removed: {initial_shape[0] - final_shape[0]}")
    
# Clean data from low quality data discovered in previous steps
# print("="*60)
# print("DATA CLEANING PROCESS")
# print("="*60)
# print(f"Starting dataset shape: {df.shape}")

# # 1. Remove exact duplicate rows
# print(f"\n1. Removing {df.duplicated().sum()} duplicate rows...")
# df = df.drop_duplicates().reset_index(drop=True)
# print(f"Shape after removing duplicates: {df.shape}")

# 2. Handle invalid ages (if any found in quality check)
# invalid_ages_before = len(df[(df['age'] < 18) | (df['age'] > 100)])
# if invalid_ages_before > 0:
#     print(f"\n2. Fixing {invalid_ages_before} invalid ages...")
#     df['age'] = df['age'].clip(lower=18, upper=100)
#     print(f"Ages capped to range [18, 100]")

# 3. Handle extreme income delta percentages
extreme_income_before = len(df[(df['income_delta_percentage'] < -100) | (df['income_delta_percentage'] > 500)])
if extreme_income_before > 0:
    print(f"\n3. Fixing {extreme_income_before} extreme income delta percentages...")
    df['income_delta_percentage'] = df['income_delta_percentage'].clip(lower=-100, upper=500)
    print(f"Income delta percentage capped to range [-100, 500]")

# 4. Handle records where due_principal > credit_limit (business logic violation)
high_due_before = len(df[df['due_principal'] > df['credit_limit']])
if high_due_before > 0:
    print(f"\n4. Fixing {high_due_before} records where due_principal > credit_limit...")
    # Cap due_principal to credit_limit
    df.loc[df['due_principal'] > df['credit_limit'], 'due_principal'] = df['credit_limit']
    print(f"Due principal capped to credit limit for affected records")

# 5. Fill missing categorical values with 'Unknown' (if any missing values found)
categorical_cols = ['marital_status', 'jobtitle_category', 'address_category', 'gender']
for col in categorical_cols:
    if col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            print(f"\n5. Filling {missing_count} missing values in {col} with 'Unknown'...")
            df[col] = df[col].fillna('Unknown')

# 6. Handle missing numerical values with median imputation
numerical_cols = ['income_delta_percentage', 'age']
for col in numerical_cols:
    if col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            median_val = df[col].median()
            print(f"\n6. Filling {missing_count} missing values in {col} with median ({median_val:.2f})...")
            df[col] = df[col].fillna(median_val)

# 7. Remove outliers (values beyond 3 standard deviations) for key numerical variables
outlier_cols = ['income_delta_percentage']
for col in outlier_cols:
    if col in df.columns:
        mean_val = df[col].mean()
        std_val = df[col].std()
        outlier_mask = (df[col] < mean_val - 3*std_val) | (df[col] > mean_val + 3*std_val)
        outliers_count = outlier_mask.sum()
        if outliers_count > 0:
            print(f"\n7. Removing {outliers_count} outliers from {col}...")
            df = df[~outlier_mask].reset_index(drop=True)

print(f"\n" + "="*40)
print("CLEANING SUMMARY")
print("="*40)
print(f"Final dataset shape: {df.shape}")
print(f"Records removed during cleaning: {len(df_backup) - len(df)}")
print(f"Data quality improvement completed!")