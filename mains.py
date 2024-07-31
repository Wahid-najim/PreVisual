# importing all necessarry library
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns



# Defining the Main Application Class

class DataVisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data PreproVisualiz")

        self.upload_button = tk.Button(root, text="Upload Data File", command=self.upload_file)
        self.upload_button.pack(pady=20)

        self.visualize_button = tk.Button(root, text="Visualize Data", command=self.visualize_data, state=tk.DISABLED)
        self.visualize_button.pack(pady=20)

        self.data = None
        
        
    
    
    # Handling File Upload

    def upload_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.data = pd.read_csv(file_path)
            self.preprocess_data()
            self.visualize_button.config(state=tk.NORMAL)
    
    
    # Preprocessing the Data
    def preprocess_data(self):
        if self.data is not None:
            # Separate numerical and categorical columns
            num_cols = self.data.select_dtypes(include=[np.number]).columns
            cat_cols = self.data.select_dtypes(include=[object]).columns
            
            # Handle missing values
            # Numerical columns: Impute using mean
            num_imputer = SimpleImputer(strategy='mean')
            self.data[num_cols] = num_imputer.fit_transform(self.data[num_cols])

            # Categorical columns: Impute using the most frequent value
            cat_imputer = SimpleImputer(strategy='most_frequent')
            self.data[cat_cols] = cat_imputer.fit_transform(self.data[cat_cols])
            
            # Encode categorical columns
            label_encoders = {}
            for col in cat_cols:
                le = LabelEncoder()
                self.data[col] = le.fit_transform(self.data[col])
                label_encoders[col] = le

            # Handle imbalanced data by upsampling the minority class
            # Assuming the last column is the target variable
            target_col = self.data.columns[-1]
            majority_class = self.data[self.data[target_col] == self.data[target_col].mode()[0]]
            minority_class = self.data[self.data[target_col] != self.data[target_col].mode()[0]]
            minority_upsampled = resample(minority_class, replace=True, n_samples=len(majority_class), random_state=42)
            self.data = pd.concat([majority_class, minority_upsampled])

            # Data reduction (removing duplicates)
            self.data = self.data.drop_duplicates()

            # Feature Engineering (example: create a new feature)
            self.data['new_feature'] = self.data.iloc[:, 0] * 2  # Example feature

            # Data Transformation (scaling)
            scaler = StandardScaler()
            features = self.data.drop(columns=[target_col])
            scaled_features = scaler.fit_transform(features)
            self.data[features.columns] = scaled_features
            
    
    
    # finally visualizing the data

    def visualize_data(self):
        if self.data is not None:
            plt.figure(figsize=(18, 12))

            # Visualization: Distribution of Features
            plt.subplot(2, 3, 1)
            sns.histplot(self.data, kde=True)
            plt.title('Feature Distribution')

            # Visualization: Correlation Heatmap
            plt.subplot(2, 3, 2)
            sns.heatmap(self.data.corr(), annot=True, cmap='coolwarm')
            plt.title('Correlation Heatmap')

            # Visualization: Pair Plot
            plt.subplot(2, 3, 3)
            sns.pairplot(self.data)
            plt.title('Pair Plot')

            # Visualization: Box Plot
            plt.subplot(2, 3, 4)
            num_cols = self.data.select_dtypes(include=[np.number]).columns
            sns.boxplot(data=self.data[num_cols])
            plt.title('Box Plot')

            # Visualization: Bar Plot (Example for the target variable)
            plt.subplot(2, 3, 5)
            if self.data[self.data.columns[-1]].dtype == np.int64 or self.data[self.data.columns[-1]].dtype == np.float64:
                sns.barplot(x=self.data[self.data.columns[-1]].value_counts().index, y=self.data[self.data.columns[-1]].value_counts())
                plt.title('Bar Plot of Target Variable')

            # Visualization: Scatter Plot (Example between the first two features)
            plt.subplot(2, 3, 6)
            if len(num_cols) >= 2:
                sns.scatterplot(x=self.data[num_cols[0]], y=self.data[num_cols[1]])
                plt.title('Scatter Plot')

            plt.tight_layout()
            plt.show()
            
            


# Running the main application

if __name__ == "__main__":
    root = tk.Tk()
    app = DataVisualizerApp(root)
    root.mainloop()
