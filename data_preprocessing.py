import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Handle data cleaning, preprocessing, and validation"""
    
    def __init__(self):
        self.original_shape = None
        self.null_stats = {}
        self.outlier_stats = {}
        
    def load_excel_data(self, file_path, sheet_name=0):
        """Load data from Excel file"""
        try:
            print(f"Loading data from: {file_path}")
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            self.original_shape = df.shape
            print(f"✓ Loaded successfully! Shape: {df.shape}")
            print(f"\nColumn Names:\n{df.columns.tolist()}\n")
            return df
        except Exception as e:
            print(f"✗ Error loading file: {str(e)}")
            raise
    
    def clean_multicolumn_structure(self, df):
        """
        Clean Excel files with multiple location columns
        Expects pattern: Location | Metric1 | Metric2 | Metric3 ...
        Returns aggregated demand data
        """
        print("\n" + "="*70)
        print("CLEANING MULTI-COLUMN STRUCTURE")
        print("="*70)
        
        df_clean = df.copy()
        
        # Get location columns (non-Unnamed columns)
        location_cols = [col for col in df_clean.columns if not col.startswith('Unnamed')]
        print(f"Found {len(location_cols)} location columns:")
        print(location_cols)
        
        # For each location, aggregate numeric data across related columns
        aggregated_data = {}
        
        for loc in location_cols:
            # Get the location column and its related unnamed columns
            loc_idx = df_clean.columns.get_loc(loc)
            
            # Collect values from this location and next 2 columns (if they're "Unnamed")
            values = []
            for i in range(3):  # Typically 3 metrics per location
                if loc_idx + i < len(df_clean.columns):
                    col = df_clean.columns[loc_idx + i]
                    if col == loc or col.startswith('Unnamed'):
                        values.extend(df_clean[col].values)
            
            # Take numeric values
            numeric_vals = [v for v in values if isinstance(v, (int, float)) and not pd.isna(v)]
            if numeric_vals:
                aggregated_data[loc] = np.mean(numeric_vals)
        
        print(f"\nAggregated demand data per location:")
        for loc, demand in aggregated_data.items():
            print(f"  {loc}: {demand:.2f}")
        
        return aggregated_data, location_cols
    
    def extract_numeric_columns(self, df):
        """Extract and aggregate numeric data from multi-location Excel"""
        print("\n" + "="*70)
        print("EXTRACTING NUMERIC DATA")
        print("="*70)
        
        df_numeric = df.select_dtypes(include=[np.number])
        
        if df_numeric.empty:
            print("✗ No numeric data found. Attempting type conversion...")
            df = df.copy()
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df_numeric = df.select_dtypes(include=[np.number])
        
        print(f"✓ Found {df_numeric.shape[1]} numeric columns")
        print(f"  Numeric columns: {df_numeric.columns.tolist()}")
        
        return df_numeric
    
    def create_timeseries_from_multicolumn(self, df, location_cols):
        """Create a time series DataFrame from multi-column structure"""
        print("\n" + "="*70)
        print("CREATING TIME SERIES DATA")
        print("="*70)
        
        # Use first column (or index) as date if available
        df_ts = df.copy()
        
        # Try to use first non-Unnamed column as date
        date_col = None
        for col in df_ts.columns:
            if not col.startswith('Unnamed'):
                try:
                    df_ts[col] = pd.to_datetime(df_ts[col])
                    date_col = col
                    break
                except:
                    pass
        
        if date_col is None:
            # Create date index
            df_ts['Date'] = pd.date_range(start='2020-01-01', periods=len(df_ts))
            date_col = 'Date'
        
        print(f"✓ Using '{date_col}' as date column")
        
        # Aggregate all numeric columns as demand
        numeric_cols = df_ts.select_dtypes(include=[np.number]).columns
        df_ts['Demand'] = df_ts[numeric_cols].sum(axis=1)
        
        df_ts = df_ts[[date_col, 'Demand']].copy()
        df_ts.columns = ['Date', 'Demand']
        
        print(f"✓ Created time series: {df_ts.shape}")
        print(f"  Date range: {df_ts['Date'].min()} to {df_ts['Date'].max()}")
        
        return df_ts
    
    def analyze_missing_values(self, df):
        """Analyze missing values"""
        print("\n" + "="*70)
        print("ANALYZING MISSING VALUES")
        print("="*70)
        
        null_counts = df.isnull().sum()
        null_percentages = (df.isnull().sum() / len(df) * 100)
        
        missing_info = pd.DataFrame({
            'Column': df.columns,
            'Null_Count': null_counts.values,
            'Null_Percentage': null_percentages.values
        })
        
        print(missing_info.to_string(index=False))
        self.null_stats = missing_info
        
        return missing_info
    
    def handle_missing_values(self, df, strategy='auto'):
        """Handle missing values intelligently"""
        print("\n" + "="*70)
        print("HANDLING MISSING VALUES")
        print("="*70)
        
        df_clean = df.copy()
        cols_with_nulls = df.columns[df.isnull().any()]
        
        for col in cols_with_nulls:
            null_count = df_clean[col].isnull().sum()
            
            if strategy == 'auto':
                if df_clean[col].dtype in ['float64', 'int64']:
                    df_clean[col] = df_clean[col].fillna(method='ffill')
                    df_clean[col] = df_clean[col].interpolate(method='linear')
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
                else:
                    df_clean[col] = df_clean[col].fillna('Unknown')
            
            print(f"✓ {col}: Handled {null_count} missing values")
        
        return df_clean
    
    def remove_dummy_values(self, df, dummy_values=None):
        """Remove dummy/placeholder values"""
        if dummy_values is None:
            dummy_values = [-1, -999, 0, 'N/A', 'NA', 'NULL']
        
        print("\n" + "="*70)
        print("REMOVING DUMMY/PLACEHOLDER VALUES")
        print("="*70)
        
        df_clean = df.copy()
        rows_before = len(df_clean)
        
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            for dummy_val in dummy_values:
                try:
                    dummy_num = float(dummy_val)
                    count = len(df_clean[df_clean[col] == dummy_num])
                    if count > 0:
                        df_clean = df_clean[df_clean[col] != dummy_num]
                        print(f"✓ Removed {count} rows with value '{dummy_val}' from '{col}'")
                except (ValueError, TypeError):
                    pass
        
        # Keep only positive demand
        if 'Demand' in df_clean.columns:
            df_clean = df_clean[df_clean['Demand'] > 0]
        
        rows_removed = rows_before - len(df_clean)
        print(f"\nTotal rows removed: {rows_removed}")
        print(f"Remaining rows: {len(df_clean)}")
        
        return df_clean
    
    def detect_outliers(self, df, numeric_columns=None, method='iqr'):
        """Detect outliers"""
        print("\n" + "="*70)
        print("DETECTING OUTLIERS")
        print("="*70)
        
        if numeric_columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        outliers_info = {}
        
        for col in numeric_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_count = len(df[(df[col] < lower_bound) | (df[col] > upper_bound)])
            
            if outlier_count > 0:
                outliers_info[col] = outlier_count
                print(f"✓ {col}: Found {outlier_count} outliers ({outlier_count/len(df)*100:.2f}%)")
        
        self.outlier_stats = outliers_info
        return outliers_info
    
    def handle_outliers(self, df, numeric_columns=None, action='cap'):
        """Handle outliers"""
        print("\n" + "="*70)
        print("HANDLING OUTLIERS")
        print("="*70)
        
        if numeric_columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        df_clean = df.copy()
        
        for col in numeric_columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            if action == 'cap':
                df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
                print(f"✓ {col}: Capped between {lower_bound:.2f} and {upper_bound:.2f}")
        
        return df_clean
    
    def remove_duplicates(self, df):
        """Remove duplicate rows"""
        print("\n" + "="*70)
        print("REMOVING DUPLICATES")
        print("="*70)
        
        rows_before = len(df)
        df_clean = df.drop_duplicates().reset_index(drop=True)
        duplicates_removed = rows_before - len(df_clean)
        
        print(f"✓ Removed {duplicates_removed} duplicate rows")
        print(f"Remaining rows: {len(df_clean)}")
        
        return df_clean
    
    def normalize_datetime(self, df, date_column='Date'):
        """Normalize datetime"""
        print("\n" + "="*70)
        print("NORMALIZING DATETIME")
        print("="*70)
        
        df_clean = df.copy()
        
        try:
            df_clean[date_column] = pd.to_datetime(df_clean[date_column])
            df_clean = df_clean.sort_values(date_column).reset_index(drop=True)
            print(f"✓ Date column normalized and sorted")
            print(f"  Date range: {df_clean[date_column].min()} to {df_clean[date_column].max()}")
            return df_clean
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            return df_clean
    
    def get_data_summary(self, df):
        """Generate data summary"""
        print("\n" + "="*70)
        print("DATA SUMMARY")
        print("="*70)
        
        print(f"\nShape: {df.shape}")
        print(f"\nData Types:\n{df.dtypes}")
        print(f"\nBasic Statistics:\n{df.describe()}")
        
        return df.describe()
    
    def full_preprocessing_pipeline(self, file_path, sheet_name=0, remove_outliers=True):
        """Execute complete preprocessing pipeline"""
        print("\n" + "█"*70)
        print("█ STARTING COMPLETE PREPROCESSING PIPELINE")
        print("█"*70)
        
        # Load data
        df = self.load_excel_data(file_path, sheet_name)
        
        # Create time series from multi-column structure
        location_cols = [col for col in df.columns if not col.startswith('Unnamed')]
        df = self.create_timeseries_from_multicolumn(df, location_cols)
        
        # Analyze
        self.analyze_missing_values(df)
        
        # Clean
        df = self.handle_missing_values(df, strategy='auto')
        df = self.remove_duplicates(df)
        df = self.remove_dummy_values(df)
        
        # Detect and handle outliers
        if remove_outliers:
            self.detect_outliers(df)
            df = self.handle_outliers(df, action='cap')
        
        # Normalize datetime
        df = self.normalize_datetime(df, 'Date')
        
        # Summary
        self.get_data_summary(df)
        
        print("\n" + "█"*70)
        print("█ PREPROCESSING COMPLETE!")
        print("█"*70)
        print(f"Original shape: {self.original_shape}")
        print(f"Final shape: {df.shape}")
        
        return df
