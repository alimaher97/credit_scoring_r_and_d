import pandas as pd
from dataclasses import dataclass

@dataclass
class CreditDatasetConfig:
    data_path: str
    filters: dict
    lowest_age: int
    highest_age: int
    max_income_delta_percentage: float
    min_income_delta_percentage: float
    categorical_cols: list
    numerical_cols: list
    outlier_cols: list
    good_buckets: list
    not_null_cols: list
    gtz_cols: list

class CreditDataPreprocessor:
    def __init__(self, data, config: CreditDatasetConfig):
        self.df_backup = data
        self.df = data.copy()
        self.documentation = {}
        self.config = config

    def __filter_categorical(self):
        """
        Removes rows from the DataFrame based on specified categorical values.
        """
        if 'filtering_steps' not in self.documentation:
            self.documentation['filtering_steps'] = []
        # Filter for UnBanked customers only
        self.df = self.df[self.df['limit_source'] == 'UnBanked'].copy()
        print(f"Original dataset shape: {self.df_backup.shape}")
        print(f"Filtered dataset shape (UnBanked only): {self.df.shape}")
        print(f"\nUnique values in limit_source (original):")
        print(self.df_backup['limit_source'].value_counts())
        # Filter for none special programmes
        self.df = self.df[self.df['special_program_flag'] != 'True'].copy()
        print("df shape:", self.df.shape)
        print("\ndf columns:", self.df.columns.tolist())
        print("\nFirst 5 rows of df:")
        print(self.df.head())
        #remove UnBanked 3 and 4
        print(self.df['rank'].value_counts().sort_index())
        print(f"\nUnbanked rank distribution:")
        print(f"Shape before removing rank 3 and 4: {self.df.shape}")
        self.df = self.df[~self.df['rank'].isin([3, 4])].copy()
        print(f"Shape after removing rank 3 and 4: {self.df.shape}")
        self.documentation['filtering_steps'].append(f"Filtered for UnBanked customers and removed special programmes and ranks 3 & 4 - New shape: {self.df.shape}")
    def __remove_null_values_rows(self):
        if 'filtering_steps' not in self.documentation:
            self.documentation['filtering_steps'] = []
        # remove customers with no credit limit or credit limit = zero
        print(f"\nUnbanked rank distribution:")
        print(f"Shape before filtering by credit limit: {self.df.shape}")
        self.df = self.df[self.df["credit_limit"].notnull() & (self.df["credit_limit"] > 0)]
        print(f"Shape after filtering by credit limit: {self.df.shape}")
        # remove customers with null customer_bucket as they might got limit but no loans
        print(f"\nShape before filtering by customer_bucket: {self.df.shape}")
        self.df = self.df[self.df["customer_bucket"].notnull()]
        print(f"Shape after filtering by customer_bucket: {self.df.shape}")
        # remove customers with CANCELLED                     1458  r
        # CANCELLED-PARTIAL-REFUND        30  r
        self.df = self.df[~self.df['customer_bucket'].isin(['CANCELLED', 'CANCELLED-PARTIAL-REFUND'])]
        self.documentation['filtering_steps'].append(f"Removed rows with null or zero credit_limit and null customer_bucket - New shape: {self.df.shape}")
    def __remove_duplicates(self):
        """
        Removes duplicate rows from the DataFrame.
        """
        if 'filtering_steps' not in self.documentation:
            self.documentation['filtering_steps'] = []
        # Remove exact duplicate rows
        print(f"\n1. Removing {self.df.duplicated().sum()} duplicate rows...")
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        print(f"Shape after removing duplicates: {self.df.shape}")
        self.documentation['filtering_steps'].append(f"Removed duplicate rows - New shape: {self.df.shape}")
    def __handle_invalid_ages(self):
        """
        Removes rows with invalid ages (less than lowest_age or greater than highest_age).
        """
        if 'filtering_steps' not in self.documentation:
            self.documentation['filtering_steps'] = []
        # Handle invalid ages (if any found in quality check)
        invalid_ages_before = len(self.df[(self.df['age'] < 18) | (self.df['age'] > 100)])
        if invalid_ages_before > 0:
            print(f"\n2. Fixing {invalid_ages_before} invalid ages...")
            self.df['age'] = self.df['age'].clip(lower=18, upper=100)
            print(f"Ages capped to range [18, 100]")
        self.documentation['filtering_steps'].append(f"Handled invalid ages - Ages capped to range [{self.config.lowest_age}, {self.config.highest_age}]")
    def __handle_extreme_income_delta(self):
        """
        Removes rows with extreme income delta percentages (less than min_income_delta_percentage or greater than max_income_delta_percentage).
        """
        if 'filtering_steps' not in self.documentation:
            self.documentation['filtering_steps'] = []
        # Handle extreme income delta percentages
        extreme_income_before = len(self.df[(self.df['income_delta_percentage'] < -100) | (self.df['income_delta_percentage'] > 500)])
        if extreme_income_before > 0:
            print(f"\n3. Fixing {extreme_income_before} extreme income delta percentages...")
            self.df['income_delta_percentage'] = self.df['income_delta_percentage'].clip(lower=-100, upper=500)
        self.documentation['filtering_steps'].append(f"Handled extreme income delta percentages - Capped to range [{self.config.min_income_delta_percentage}, {self.config.max_income_delta_percentage}]")
    def __handle_due_principal_vs_credit_limit(self):
        """
        Caps due_principal to credit_limit if due_principal exceeds credit_limit.
        """
        if 'filtering_steps' not in self.documentation:
            self.documentation['filtering_steps'] = []
        # Handle records where due_principal > credit_limit (business logic violation)
        high_due_before = len(self.df[self.df['due_principal'] > self.df['credit_limit']])
        if high_due_before > 0:
            print(f"\n4. Fixing {high_due_before} records where due_principal > credit_limit...")
            # Cap due_principal to credit_limit
            self.df.loc[self.df['due_principal'] > self.df['credit_limit'], 'due_principal'] = self.df['credit_limit']
            print(f"Due principal capped to credit limit for affected records")
        self.documentation['filtering_steps'].append(f"Handled due_principal vs credit_limit - Capped due_principal to credit_limit where necessary")
    def __missing_categorical_values_imputation(self):
        """
        Imputes missing values in categorical columns with 'Unknown'.
        """
        if 'filtering_steps' not in self.documentation:
            self.documentation['filtering_steps'] = []
        # Fill missing categorical values with 'Unknown' (if any missing values found)
        categorical_cols = ['marital_status', 'jobtitle_category', 'address_category', 'gender']
        for col in categorical_cols:
            if col in self.df.columns:
                missing_count = self.df[col].isnull().sum()
                if missing_count > 0:
                    print(f"\n5. Filling {missing_count} missing values in {col} with 'Unknown'...")
                    self.df[col] = self.df[col].fillna('Unknown')
        self.documentation['filtering_steps'].append(f"Imputed missing categorical values with 'Unknown'")
    def __missing_numerical_values_imputation(self):
        """
        Imputes missing values in numerical columns with the median of the column.
        """
        if 'filtering_steps' not in self.documentation:
            self.documentation['filtering_steps'] = []
        # Handle missing numerical values with median imputation
        numerical_cols = ['income_delta_percentage', 'age']
        for col in numerical_cols:
            if col in self.df.columns:
                missing_count = self.df[col].isnull().sum()
                if missing_count > 0:
                    median_val = self.df[col].median()
                    print(f"\n6. Filling {missing_count} missing values in {col} with median ({median_val:.2f})...")
                    self.df[col] = self.df[col].fillna(median_val)
        self.documentation['filtering_steps'].append(f"Imputed missing numerical values with median")
    def __remove_outliers(self):
        """
        Removes outliers from specified numerical columns using the 3 standard deviations rule.
        """
        if 'filtering_steps' not in self.documentation:
            self.documentation['filtering_steps'] = []
        # 7. Remove outliers (values beyond 3 standard deviations) for key numerical variables
        outlier_cols = ['income_delta_percentage']
        for col in outlier_cols:
            if col in self.df.columns:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                outlier_mask = (self.df[col] < mean_val - 3*std_val) | (self.df[col] > mean_val + 3*std_val)
                outliers_count = outlier_mask.sum()
                if outliers_count > 0:
                    print(f"\n7. Removing {outliers_count} outliers from {col}...")
                    self.df = self.df[~outlier_mask].reset_index(drop=True)
        self.documentation['filtering_steps'].append(f"Removed outliers from specified numerical columns using 3 standard deviations rule")
    def __create_target_variable(self):
        """
        Creates a binary target variable based on customer_bucket values.
        Good customers (0) are those in specified good_buckets, bad customers (1)
        """
        good_buckets = ['CURRENT', 'SETTLED', 'SETTLED-PAIDOFF', 'BUCKET-1', 'BUCKET-2']
        
        self.df['target'] = (~self.df['customer_bucket'].isin(good_buckets)).astype(int)

        print("Customer bucket distribution:")
        print(self.df['customer_bucket'].value_counts())

        print(f"\nTarget variable distribution:")
        print(self.df['target'].value_counts())
        print(f"Good customers (target=0): {(self.df['target'] == 0).sum()} ({(self.df['target'] == 0).mean()*100:.2f}%)")
        print(f"Bad customers (target=1): {(self.df['target'] == 1).sum()} ({(self.df['target'] == 1).mean()*100:.2f}%)")

        print(f"\nGood customer buckets: {good_buckets}")
        print(f"Bad customer buckets: {self.df[self.df['target'] == 1]['customer_bucket'].unique().tolist()}")
    def __create_final_features_set(self):
        """
        Selects and returns the final set of features for modeling.
        """
        # Create final features dataframe with target variable
        self.df = self.df[['customer_id', 'age', 'marital_status', 'income_delta_percentage', 
                    'address_category', 'jobtitle_category', 'gender', 'target']].copy()

        print("Final dataset shape:", self.df.shape)
        print("\nFinal dataset columns:", self.df.columns.tolist())
        print("\nFirst 5 rows of final dataset:")
        print(self.df.head())
        print(f"\nTarget distribution:")
        print(self.df['target'].value_counts())
        print(f"Good rate: {(self.df['target'] == 0).mean()*100:.2f}%")
        print(f"Bad rate: {(self.df['target'] == 1).mean()*100:.2f}%")
    def __cleaning_summary(self):
        if 'cleaning_summary' not in self.documentation:
            self.documentation['cleaning_summary'] = []
        self.documentation['cleaning_summary'].append(f"Final dataset shape: {self.df.shape}\n")
        self.documentation['cleaning_summary'].append(f"Records removed during cleaning: {self.df_backup.shape[0] - self.df.shape[0]}\n")
        self.documentation['cleaning_summary'].append(f"Data quality improvement completed!\n")
    def __final_summary(self):
        if 'target_definition' not in self.documentation:
            self.documentation['target_definition'] = None
        if 'features_description' not in self.documentation:
            self.documentation['features_description'] = None
        if 'dataset_info' not in self.documentation:
            self.documentation['dataset_info'] = None
        self.documentation['target_definition'] = {
        "good_customers_buckets": ['CURRENT', 'SETTLED', 'CANCELLED', 'CANCELLED-PARTIAL-REFUND', 
                                  'SETTLE-RESCHEDULED', 'BUCKET-1', 'BUCKET-2'],
        "bad_customers_buckets": "All other customer_bucket values",
        "target_encoding": "0 = Good customer, 1 = Bad customer"
    }
        self.documentation['features_description'] = {
        "customer_id": "Unique customer identifier",
        "age": "Customer age (capped between 18-100)",
        "marital_status": "Customer marital status",
        "income_delta_percentage": "Income change percentage (capped between -100% to 500%)",
        "address_category": "Address quality category (A-E)",
        "jobtitle_category": "Job title quality category (A-E)", 
        "gender": "Customer gender (M/F)",
        "target": "Binary target variable (0=Good, 1=Bad)"
    }
        self.documentation['dataset_info'] = {
        "original_shape": self.df_backup.shape,
        "final_shape": self.df.shape,
        "records_removed":  self.df_backup.shape[0] - self.df.shape[0],
        "unique_customers": self.df['customer_id'].nunique(),
        "target_distribution": {
            "good_customers": int((self.df['target'] == 0).sum()),
            "bad_customers": int((self.df['target'] == 1).sum()),
            "good_rate_percent": round((self.df['target'] == 0).mean()*100, 2),
            "bad_rate_percent": round((self.df['target'] == 1).mean()*100, 2)
        }
    }
        
    def preprocess(self):
        self.__remove_null_values_rows()
        self.__filter_categorical()
        self.__remove_duplicates()
        self.__handle_invalid_ages()
        self.__handle_extreme_income_delta()
        self.__handle_due_principal_vs_credit_limit()
        self.__missing_categorical_values_imputation()
        self.__missing_numerical_values_imputation()
        self.__remove_outliers()
        self.__create_target_variable()
        self.__create_final_features_set()
        self.__cleaning_summary()
        self.__final_summary()
        return self.df, self.documentation