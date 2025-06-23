import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Titanic-Dataset.csv")

# STEP 1:
print("First 5 rows of the dataset : ")
print(df.head())
print("Dataset information : ")
print(df.info())
print("Missing values : ")
print(df.isnull().sum())

# STEP 2:
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)
print("Updated : ")
print(df.isnull().sum())

# STEP 3
df.drop(columns=['Name', 'Ticket'], inplace=True)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True) 
print("Updated : ")
print(df.head())

#STEP 4
scaler = StandardScaler()
num_cols = ['Age', 'Fare', 'SibSp', 'Parch']
df[num_cols] = scaler.fit_transform(df[num_cols])
print("Updated : ")
print(df.head())

#STEP 5
plt.figure(figsize=(12, 8))
for i, col in enumerate(num_cols):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()

# Step 8: Remove Outliers using IQR
def remove_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

df_cleaned = remove_outliers_iqr(df, num_cols)

print("Previous : ", df.shape)
print("Updated : ", df_cleaned.shape)

df_cleaned.to_csv("Titanic-Dataset-Cleaned.csv", index=False)
print("The Cleaned and Updated Dataset is saved by the name : 'Titanic-Dataset-Cleaned.csv'")
