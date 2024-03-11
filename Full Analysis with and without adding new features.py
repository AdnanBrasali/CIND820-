#!/usr/bin/env python
# coding: utf-8

# In[ ]:


################################################################################################## 


# In[ ]:


################################   MODELING BEFORE ADDING NEW FEATURES  ########################## 


# In[ ]:


##################################################################################################


# In[139]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
dataset_path = 'C:/Users/adnan/OneDrive/Desktop/CIND820 capstone/dataset.csv'
df = pd.read_csv(dataset_path)
print(df.head(10))


# In[140]:


# Convert columns with dollar sign to numeric and Date column to datetime
cols_with_dollar = ['Close/Last', 'Open', 'High', 'Low']
df[cols_with_dollar] = df[cols_with_dollar].replace({'\$': '', ',': ''}, regex=True).astype(float)
def parse_mixed_dates(date_str):
    
    try:
        return pd.to_datetime(date_str, format='%m/%d/%Y')
    except ValueError:
        
        try:
            return pd.to_datetime(date_str, format='%d-%m-%Y')
        except ValueError:
            
            return pd.NaT

df['Date'] = df['Date'].apply(parse_mixed_dates)

print(df.head(10))
# Descriptive Statistics
descriptive_stats = df.describe()
print(descriptive_stats)
# Specify your path
path = 'C:/Users/adnan/OneDrive/Desktop/CIND820 capstone/'
#  save the descriptive statistics
descriptive_stats.to_csv('descriptive_stats.csv')
descriptive_stats.to_excel('descriptive_stats.xlsx')
descriptive_stats.to_latex('descriptive_stats.tex')


# In[141]:


# Separate the dataframe into different dataframes based on scale
df_prices = df[['Close/Last', 'Open', 'High', 'Low']]
df_volume = df[['Volume']]  

# Plotting the prices
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_prices)
plt.title('Boxplot of Price Features')
plt.show()


# In[142]:


# Plotting the volume
plt.figure(figsize=(5, 5))
sns.boxplot(data=df_volume)
plt.title('Boxplot of Volume Feature')
plt.show()


# In[143]:


# Remove the outliers 
columns_to_check = ['Close/Last', 'Volume', 'Open', 'High', 'Low'] 
df_filtered = df.copy()
outliers_removed = {} 

for column in columns_to_check:
    Q1 = df_filtered[column].quantile(0.25)
    Q3 = df_filtered[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter the DataFrame 
    outliers_removed[column] = len(df_filtered) - len(df_filtered[(df_filtered[column] >= lower_bound) & (df_filtered[column] <= upper_bound)])

    df_filtered = df_filtered[(df_filtered[column] >= lower_bound) & (df_filtered[column] <= upper_bound)]
total_outliers_removed = sum(outliers_removed.values())
print(f"Total number of outliers removed: {total_outliers_removed}")

for column, count in outliers_removed.items():
    print(f"Number of outliers removed in {column}: {count}")


# In[144]:


#Feature Engineering

df = df_filtered.copy()
df.loc[:, 'Price Change'] = df['Close/Last'].diff()
df.loc[:, 'Target'] = df['Price Change'].apply(lambda x: 1 if x > 0 else 0)
X = df[['Volume', 'Open', 'High', 'Low']]  
y_regression = df['Close/Last'] # Regression target
y_classification = df['Target'] # Classification target



# In[145]:


from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
class_counts = df['Target'].value_counts()

print("Class Counts:")
print(class_counts)

imbalance_ratio = class_counts.min() / class_counts.max()

print("\nClass Imbalance Ratio:", imbalance_ratio)

class_counts = df['Target'].value_counts()

plt.figure(figsize=(8, 6))
sns.barplot(x=class_counts.index, y=class_counts.values)
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(range(len(class_counts.index)), ['Class 0', 'Class 1'])  # Assuming binary classification
plt.show()


# In[146]:


df_combined = pd.concat([X, y_classification], axis=1)
# Separate majority and minority classes
df_majority = df_combined[df_combined['Target'] == 0]
df_minority = df_combined[df_combined['Target'] == 1]

# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,   
                                 n_samples=len(df_majority),   
                                 random_state=123) 

# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])


class_counts_upsampled = df_upsampled['Target'].value_counts()
print("Class Counts after Upsampling:")
print(class_counts_upsampled)
X_upsampled = df_upsampled[['Volume', 'Open', 'High', 'Low']]
y_classification_upsampled = df_upsampled['Target']
plt.figure(figsize=(8, 6))
sns.barplot(x=class_counts_upsampled.index, y=class_counts_upsampled.values)
plt.title('Class Distribution after Upsampling')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(range(len(class_counts_upsampled.index)), class_counts_upsampled.index)
plt.show()


# In[147]:


#Model Building and Evaluation

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_regression, test_size=0.2, random_state=42)

X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_upsampled, y_classification_upsampled, test_size=0.2, random_state=42)

scaler_reg = StandardScaler()
scaler_class = StandardScaler()

scaler_reg.fit(X_train_reg)
scaler_class.fit(X_train_class)

X_train_reg = scaler_reg.transform(X_train_reg)
X_train_class = scaler_class.transform(X_train_class)

X_test_reg = scaler_reg.transform(X_test_reg)
X_test_class = scaler_class.transform(X_test_class)


# In[148]:


feature_names = ['Volume', 'Open', 'High', 'Low']

X_train_reg_df = pd.DataFrame(X_train_reg, columns=feature_names)

print(X_train_reg_df.head())

X_test_class_df = pd.DataFrame(X_test_class, columns=feature_names)

print(X_test_class_df.head())


# In[149]:


from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import numpy as np

reg_models = [RandomForestRegressor(), DecisionTreeRegressor(), KNeighborsRegressor()]


class_models = [RandomForestClassifier(), DecisionTreeClassifier(), KNeighborsClassifier()]


print("Regression Models Evaluation:")
for model in reg_models:
    cv_results = cross_validate(model, X_train_reg, y_train_reg, cv=5,
                                scoring={'MAE': 'neg_mean_absolute_error', 
                                         'RMSE': 'neg_root_mean_squared_error'},
                                return_train_score=False)
    mae = -cv_results['test_MAE'].mean()
    rmse = -np.sqrt(-cv_results['test_RMSE'].mean())  
    print(f"{model.__class__.__name__}: MAE = {mae:.4f}, RMSE = {rmse:.4f}")

print("\nClassification Models Evaluation:")
for model in class_models:

    y_pred = cross_val_predict(model, X_train_class, y_train_class, cv=5)
    
    
    accuracy = accuracy_score(y_train_class, y_pred)
    precision = precision_score(y_train_class, y_pred, average='macro')  
    recall = recall_score(y_train_class, y_pred, average='macro')
    f1 = f1_score(y_train_class, y_pred, average='macro')
    cm = confusion_matrix(y_train_class, y_pred)
    
    
    print(f"{model.__class__.__name__}:")
    print(f"Accuracy = {accuracy:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, F1-Score = {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print()  
    


# In[150]:


# Classification Models Evaluation with ROC Curve

from sklearn.metrics import roc_curve, auc

print("\nClassification Models Evaluation with ROC Curve:")
class_models = [RandomForestClassifier(), DecisionTreeClassifier(), KNeighborsClassifier()]

roc_curves = []
auc_scores = []

plt.figure(figsize=(8, 6))
for model in class_models:
    y_proba = cross_val_predict(model, X_train_class, y_train_class, cv=5, method='predict_proba')
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_train_class, y_proba[:, 1])  
    
    # Calculate AUC score
    auc_score = auc(fpr, tpr)
    
    # Store ROC curve and AUC score
    roc_curves.append((fpr, tpr))
    auc_scores.append(auc_score)
    
    # Plot ROC curve
    plt.plot(fpr, tpr, label=f'{model.__class__.__name__} (AUC = {auc_score:.2f})')

plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# Print AUC scores
for model, auc_score in zip(class_models, auc_scores):
    print(f"{model.__class__.__name__}: AUC = {auc_score:.4f}")
    print()


# In[151]:


import matplotlib.pyplot as plt

reg_model = RandomForestRegressor()

reg_model.fit(X_train_reg, y_train_reg)

y_pred_reg = reg_model.predict(X_test_reg)

mae = mean_absolute_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))

plt.figure(figsize=(10, 6))
plt.scatter(y_test_reg, y_pred_reg, color='blue', alpha=0.5)
plt.plot([min(y_test_reg), max(y_test_reg)], [min(y_test_reg), max(y_test_reg)], linestyle='--', color='red')
plt.title('Actual vs. Predicted (Regression)')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

print(f"Mean Absolute Error: {mae:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")


# In[152]:


# #Perform Feature Importance Analysis All features 

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
feature_names = X.columns

# I use RandomForest models for feature importance analysis
rf_regressor = RandomForestRegressor(random_state=42)
rf_classifier = RandomForestClassifier(random_state=42)


rf_regressor.fit(X_train_reg, y_train_reg)
rf_classifier.fit(X_train_class, y_train_class)

importances_reg = rf_regressor.feature_importances_
importances_class = rf_classifier.feature_importances_

top_n_idx_reg = np.argsort(importances_reg)[-3:]
top_n_features_reg = feature_names[top_n_idx_reg]

top_n_idx_class = np.argsort(importances_class)[-3:]  
top_n_features_class = feature_names[top_n_idx_class]

print("Selected Top Features for Regression:", top_n_features_reg)
print("Selected Top Features for Classification:", top_n_features_class)


# In[153]:


# Evaluation with selected features

print("\nRegression Models Evaluation with Selected Features:")
reg_models = [RandomForestRegressor(), DecisionTreeRegressor(), KNeighborsRegressor()]

top_n_features_reg_indices = [X.columns.get_loc(feature_name) for feature_name in top_n_features_reg]

X_train_reg_selected = X_train_reg[:, top_n_features_reg_indices]
X_test_reg_selected = X_test_reg[:, top_n_features_reg_indices]

for model in reg_models:
    cv_results = cross_validate(model, X_train_reg_selected, y_train_reg, cv=5,
                                scoring={'MAE': 'neg_mean_absolute_error', 
                                         'RMSE': 'neg_root_mean_squared_error'},
                                return_train_score=False)
    mae = -cv_results['test_MAE'].mean()
    rmse = -np.sqrt(-cv_results['test_RMSE'].mean())
    print(f"{model.__class__.__name__}: MAE = {mae:.4f}, RMSE = {rmse:.4f}")


print("\nClassification Models Evaluation with Selected Features:")
class_models = [RandomForestClassifier(), DecisionTreeClassifier(), KNeighborsClassifier()]
for model in class_models:
    X_train_selected = X_train_class[:, top_n_idx_class]
    y_pred = cross_val_predict(model, X_train_selected, y_train_class, cv=5)
    
    accuracy = accuracy_score(y_train_class, y_pred)
    precision = precision_score(y_train_class, y_pred, average='macro')
    recall = recall_score(y_train_class, y_pred, average='macro')
    f1 = f1_score(y_train_class, y_pred, average='macro')
    cm = confusion_matrix(y_train_class, y_pred)
    
    print(f"{model.__class__.__name__}: Accuracy = {accuracy:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, F1-Score = {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print()
    


# In[154]:


# Classification Models Evaluation with Selected Features and ROC Curve

from sklearn.metrics import roc_curve, auc

print("\nClassification Models Evaluation with Selected Features:")
class_models = [RandomForestClassifier(), DecisionTreeClassifier(), KNeighborsClassifier()]

roc_curves = []
auc_scores = []

plt.figure(figsize=(8, 6))
for model in class_models:
    X_train_selected = X_train_class[:, top_n_idx_class]
    y_proba = cross_val_predict(model, X_train_selected, y_train_class, cv=5, method='predict_proba')
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_train_class, y_proba[:, 1])
    
    # Calculate AUC score
    auc_score = auc(fpr, tpr)
    
    # Store ROC curve and AUC score
    roc_curves.append((fpr, tpr))
    auc_scores.append(auc_score)
    
    # Plot ROC curve
    plt.plot(fpr, tpr, label=f'{model.__class__.__name__} (AUC = {auc_score:.2f})')

plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# Print AUC scores
for model, auc_score in zip(class_models, auc_scores):
    print(f"{model.__class__.__name__}: AUC = {auc_score:.4f}")
    print()


# In[155]:


import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importances(importances, features, title):
    indices = np.argsort(importances)
    sorted_features = [features[i] for i in indices]
    
    plt.figure(figsize=(10, 3))
    plt.title(title)
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), sorted_features)
    plt.xlabel('Relative Importance')
    plt.show()

plot_feature_importances(importances_reg, X.columns, "Feature Importances for Regression")
plot_feature_importances(importances_class, X.columns, "Feature Importances for Classification")


# In[156]:


from sklearn.metrics import mean_absolute_error, mean_squared_error

reg_model = RandomForestRegressor(random_state=42)

reg_model.fit(X_train_reg_selected, y_train_reg)

y_pred_reg = reg_model.predict(X_test_reg_selected)

mae = mean_absolute_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))

plt.figure(figsize=(10, 6))
plt.scatter(y_test_reg, y_pred_reg, color='blue', alpha=0.5)
plt.plot([min(y_test_reg), max(y_test_reg)], [min(y_test_reg), max(y_test_reg)], linestyle='--', color='red')
plt.title('Actual vs. Predicted (Regression)')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

print(f"Mean Absolute Error: {mae:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


##############################################################################


# In[ ]:


############### MODELING WITH ADDING NEW FEATURES   ##########################


# In[ ]:


##############################################################################


# In[157]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
dataset_path = 'C:/Users/adnan/OneDrive/Desktop/CIND820 capstone/dataset.csv'
df = pd.read_csv(dataset_path)
print(df.head(10))


# In[158]:


# Convert columns with dollar sign to numeric and Date column to datetime

cols_with_dollar = ['Close/Last', 'Open', 'High', 'Low']
df[cols_with_dollar] = df[cols_with_dollar].replace({'\$': '', ',': ''}, regex=True).astype(float)
def parse_mixed_dates(date_str):
    
    try:
        return pd.to_datetime(date_str, format='%m/%d/%Y')
    except ValueError:
        
        try:
            return pd.to_datetime(date_str, format='%d-%m-%Y')
        except ValueError:
            
            return pd.NaT


df['Date'] = df['Date'].apply(parse_mixed_dates)

print(df.head(10))

descriptive_stats = df.describe()
print(descriptive_stats)

path = 'C:/Users/adnan/OneDrive/Desktop/CIND820 capstone/'

descriptive_stats.to_csv('descriptive_stats.csv')
descriptive_stats.to_excel('descriptive_stats.xlsx')
descriptive_stats.to_latex('descriptive_stats.tex')



# In[159]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier


def calculate_momentum(series, period=14):
    momentum = series.diff(period)
    return momentum

def calculate_stochastic_oscillator(df, high_col='High', low_col='Low', close_col='Close/Last', period=14, smooth_k=3, smooth_d=3):
   
    # Calculate the highest high and lowest low over the period
    
    df['Highest High'] = df[high_col].rolling(window=period).max()
    df['Lowest Low'] = df[low_col].rolling(window=period).min()
    
 
    df['%K'] = ((df[close_col] - df['Lowest Low']) / (df['Highest High'] - df['Lowest Low'])) * 100
    
    
    df['%D'] = df['%K'].rolling(window=smooth_k).mean().rolling(window=smooth_d).mean()
    
    
    df.drop(['Highest High', 'Lowest Low'], axis=1, inplace=True)
    
    return df[['%K', '%D']]


df['Momentum'] = calculate_momentum(df['Close/Last'])

stochastic_oscillator = calculate_stochastic_oscillator(df)

df.dropna(inplace=True)

print(df.head(5))


# In[160]:


# Boxplot of Price features


df_prices = df[['Close/Last', 'Open', 'High', 'Low','Momentum','%K','%D']]  
df_volume = df[['Volume']]

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_prices)
plt.title('Boxplot of Price Features')
plt.show()

plt.figure(figsize=(5, 5))
sns.boxplot(data=df_volume)
plt.title('Boxplot of Volume Feature')
plt.show()


# In[161]:


# To remove the outliers 

columns_to_check = ['Close/Last', 'Volume','Open', 'High', 'Low','Momentum','%K','%D']  
df_filtered = df.copy()
outliers_removed = {}  

for column in columns_to_check:
    Q1 = df_filtered[column].quantile(0.25)
    Q3 = df_filtered[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_removed[column] = len(df_filtered) - len(df_filtered[(df_filtered[column] >= lower_bound) & (df_filtered[column] <= upper_bound)])

    df_filtered = df_filtered[(df_filtered[column] >= lower_bound) & (df_filtered[column] <= upper_bound)]

total_outliers_removed = sum(outliers_removed.values())
print(f"Total number of outliers removed: {total_outliers_removed}")

for column, count in outliers_removed.items():
    print(f"Number of outliers removed in {column}: {count}")


# In[162]:


#Feature Engineering

df = df_filtered.copy()


df.loc[:, 'Price Change'] = df['Close/Last'].diff()
df.loc[:, 'Target'] = df['Price Change'].apply(lambda x: 1 if x > 0 else 0)


X = df[['Volume', 'Open', 'High', 'Low','Momentum','%K','%D']]  
y_regression = df['Close/Last']            # Regression target
y_classification = df['Target']            # Classification target
X


# In[163]:


### Balance the classes in the target variable 

class_counts = df['Target'].value_counts()

print("Class Counts:")
print(class_counts)

imbalance_ratio = class_counts.min() / class_counts.max()

print("\nClass Imbalance Ratio:", imbalance_ratio)


import seaborn as sns

class_counts = df['Target'].value_counts()

plt.figure(figsize=(8, 6))
sns.barplot(x=class_counts.index, y=class_counts.values)
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(range(len(class_counts.index)), ['Class 0', 'Class 1'])
plt.show()


# In[164]:


from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns

df_combined = pd.concat([X, y_classification], axis=1)

df_majority = df_combined[df_combined['Target'] == 0]
df_minority = df_combined[df_combined['Target'] == 1]

df_minority_upsampled = resample(df_minority, 
                                 replace=True,   
                                 n_samples=len(df_majority),    
                                 random_state=123) 

df_upsampled = pd.concat([df_majority, df_minority_upsampled])

class_counts_upsampled = df_upsampled['Target'].value_counts()
print("Class Counts after Upsampling:")
print(class_counts_upsampled)

X_upsampled = df_upsampled[['Volume', 'Open', 'High', 'Low','Momentum','%K','%D']]
y_classification_upsampled = df_upsampled['Target']

plt.figure(figsize=(8, 6))
sns.barplot(x=class_counts_upsampled.index, y=class_counts_upsampled.values)
plt.title('Class Distribution after Upsampling')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(range(len(class_counts_upsampled.index)), class_counts_upsampled.index)
plt.show()


# In[165]:


#Model Building and Evaluation

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_regression, test_size=0.2, random_state=42)

X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_upsampled, y_classification_upsampled, test_size=0.2, random_state=42)

scaler_reg = StandardScaler()
scaler_class = StandardScaler()

scaler_reg.fit(X_train_reg)
scaler_class.fit(X_train_class)

X_train_reg = scaler_reg.transform(X_train_reg)
X_train_class = scaler_class.transform(X_train_class)

X_test_reg = scaler_reg.transform(X_test_reg)
X_test_class = scaler_class.transform(X_test_class)


# In[167]:


import pandas as pd

feature_names = ['Volume', 'Open', 'High', 'Low','Momentum','%K','%D']

X_train_reg_df = pd.DataFrame(X_train_reg, columns=feature_names)


X_test_class_df = pd.DataFrame(X_test_class, columns=feature_names)



# In[168]:


from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier


reg_models = [RandomForestRegressor(), DecisionTreeRegressor(), KNeighborsRegressor()]

class_models = [RandomForestClassifier(), DecisionTreeClassifier(), KNeighborsClassifier()]

print("Regression Models Evaluation:")
for model in reg_models:
    cv_results = cross_validate(model, X_train_reg, y_train_reg, cv=5,
                                scoring={'MAE': 'neg_mean_absolute_error', 
                                         'RMSE': 'neg_root_mean_squared_error'},
                                return_train_score=False)
    mae = -cv_results['test_MAE'].mean()
    rmse = -np.sqrt(-cv_results['test_RMSE'].mean())
    print(f"{model.__class__.__name__}: MAE = {mae:.4f}, RMSE = {rmse:.4f}")


print("\nClassification Models Evaluation:")
for model in class_models:
    y_pred = cross_val_predict(model, X_train_class, y_train_class, cv=5)
    
    
    accuracy = accuracy_score(y_train_class, y_pred)
    precision = precision_score(y_train_class, y_pred, average='macro')  
    recall = recall_score(y_train_class, y_pred, average='macro')
    f1 = f1_score(y_train_class, y_pred, average='macro')
    cm = confusion_matrix(y_train_class, y_pred)
    
    print(f"{model.__class__.__name__}:")
    print(f"Accuracy = {accuracy:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, F1-Score = {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print() 


# In[169]:


# Classification Models Evaluation with ROC Curve

from sklearn.metrics import roc_curve, auc

print("\nClassification Models Evaluation with ROC Curve:")
class_models = [RandomForestClassifier(), DecisionTreeClassifier(), KNeighborsClassifier()]

roc_curves = []
auc_scores = []

plt.figure(figsize=(8, 6))
for model in class_models:
    y_proba = cross_val_predict(model, X_train_class, y_train_class, cv=5, method='predict_proba')
    
    fpr, tpr, _ = roc_curve(y_train_class, y_proba[:, 1])  
    
    auc_score = auc(fpr, tpr)
    
    roc_curves.append((fpr, tpr))
    auc_scores.append(auc_score)
    
    plt.plot(fpr, tpr, label=f'{model.__class__.__name__} (AUC = {auc_score:.2f})')

plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

for model, auc_score in zip(class_models, auc_scores):
    print(f"{model.__class__.__name__}: AUC = {auc_score:.4f}")
    print()


# In[170]:


import matplotlib.pyplot as plt

reg_model = RandomForestRegressor()

reg_model.fit(X_train_reg, y_train_reg)

y_pred_reg = reg_model.predict(X_test_reg)

mae = mean_absolute_error(y_test_reg, y_pred_reg)

rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
plt.figure(figsize=(10, 6))
plt.scatter(y_test_reg, y_pred_reg, color='blue', alpha=0.5)
plt.plot([min(y_test_reg), max(y_test_reg)], [min(y_test_reg), max(y_test_reg)], linestyle='--', color='red')
plt.title('Actual vs. Predicted (Regression)')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

print(f"Mean Absolute Error: {mae:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")


# In[171]:


from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

feature_names = X.columns

rf_regressor = RandomForestRegressor(random_state=42)
rf_classifier = RandomForestClassifier(random_state=42)

rf_regressor.fit(X_train_reg, y_train_reg)
rf_classifier.fit(X_train_class, y_train_class)

importances_reg = rf_regressor.feature_importances_
importances_class = rf_classifier.feature_importances_

top_n_idx_reg = np.argsort(importances_reg)[-3:] 
top_n_features_reg = feature_names[top_n_idx_reg]

top_n_idx_class = np.argsort(importances_class)[-3:]  
top_n_features_class = feature_names[top_n_idx_class]

print("Selected Top Features for Regression:", top_n_features_reg)
print("Selected Top Features for Classification:", top_n_features_class)


# In[172]:


import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importances(importances, features, title):
    indices = np.argsort(importances)
    sorted_features = [features[i] for i in indices]
    
    plt.figure(figsize=(10, 4))
    plt.title(title)
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), sorted_features)
    plt.xlabel('Relative Importance')
    plt.show()

plot_feature_importances(importances_reg, X.columns, "Feature Importances for Regression")
plot_feature_importances(importances_class, X.columns, "Feature Importances for Classification")


# In[173]:


# Regression and classification Models Evaluation with Selected Features

print("\nRegression Models Evaluation with Selected Features:")
reg_models = [RandomForestRegressor(), DecisionTreeRegressor(), KNeighborsRegressor()]

top_n_features_reg_indices = [X.columns.get_loc(feature_name) for feature_name in top_n_features_reg]

X_train_reg_selected = X_train_reg[:, top_n_features_reg_indices]
X_test_reg_selected = X_test_reg[:, top_n_features_reg_indices]


for model in reg_models:
    cv_results = cross_validate(model, X_train_reg_selected, y_train_reg, cv=5,
                                scoring={'MAE': 'neg_mean_absolute_error', 
                                         'RMSE': 'neg_root_mean_squared_error'},
                                return_train_score=False)
    mae = -cv_results['test_MAE'].mean()
    rmse = -np.sqrt(-cv_results['test_RMSE'].mean())
    
    print(f"{model.__class__.__name__}: MAE = {mae:.4f}, RMSE = {rmse:.4f}")


print("\nClassification Models Evaluation with Selected Features:")
class_models = [RandomForestClassifier(), DecisionTreeClassifier(), KNeighborsClassifier()]
for model in class_models:
    X_train_selected = X_train_class[:, top_n_idx_class]
    y_pred = cross_val_predict(model, X_train_selected, y_train_class, cv=5)
    
    accuracy = accuracy_score(y_train_class, y_pred)
    precision = precision_score(y_train_class, y_pred, average='macro')
    recall = recall_score(y_train_class, y_pred, average='macro')
    f1 = f1_score(y_train_class, y_pred, average='macro')
    cm = confusion_matrix(y_train_class, y_pred)
    
    print(f"{model.__class__.__name__}: Accuracy = {accuracy:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, F1-Score = {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print()


# In[174]:


# Classification Models Evaluation with Selected Features and ROC Curve

from sklearn.metrics import roc_curve, auc

print("\nClassification Models Evaluation with Selected Features:")

class_models = [RandomForestClassifier(), DecisionTreeClassifier(), KNeighborsClassifier()]

roc_curves = []
auc_scores = []

plt.figure(figsize=(8, 6))
for model in class_models:
    X_train_selected = X_train_class[:, top_n_idx_class]
    y_proba = cross_val_predict(model, X_train_selected, y_train_class, cv=5, method='predict_proba')
    
    fpr, tpr, _ = roc_curve(y_train_class, y_proba[:, 1])  
    
    auc_score = auc(fpr, tpr)
    
    roc_curves.append((fpr, tpr))
    auc_scores.append(auc_score)
    
    plt.plot(fpr, tpr, label=f'{model.__class__.__name__} (AUC = {auc_score:.2f})')

plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

for model, auc_score in zip(class_models, auc_scores):
    print(f"{model.__class__.__name__}: AUC = {auc_score:.4f}")
    print()


# In[175]:


# Plot actual vs. predicted for regression

from sklearn.metrics import mean_absolute_error, mean_squared_error

reg_model = RandomForestRegressor(random_state=42)

reg_model.fit(X_train_reg_selected, y_train_reg)

y_pred_reg = reg_model.predict(X_test_reg_selected)

mae = mean_absolute_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))


plt.figure(figsize=(10, 6))
plt.scatter(y_test_reg, y_pred_reg, color='blue', alpha=0.5)
plt.plot([min(y_test_reg), max(y_test_reg)], [min(y_test_reg), max(y_test_reg)], linestyle='--', color='red')
plt.title('Actual vs. Predicted (Regression)')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

print(f"Mean Absolute Error: {mae:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")


# In[ ]:




