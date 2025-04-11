import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

# Load the dataset
df=pd.read_csv("/Users/sujalkarki/Desktop/Mobile Project /Mobiles.csv",encoding='latin1')
print(df.head(5)) #display the first 5 rows
print(df.columns)  #display the columns of the dataset
df.info()   # Get the summary of the dataset

# Change the value type of some columns from string to float
numeric_columns = ['Mobile Weight', 'RAM', 'Front Camera', 'Back Camera', 'Battery Capacity', 
                   'Launched Price (Pakistan)', 'Launched Price (India)', 'Launched Price (China)', 'Launched Price (USA)', 
                   'Launched Price (Dubai)']

for col in numeric_columns:
    df[col] = df[col].str.replace(r'[^\d]', '', regex=True)
    df[col] = df[col].replace('', 'NaN')
    df[col] = df[col].astype(float)
df.replace([np.inf, -np.inf], np.nan, inplace=True)

expressions = ['inches','(main)','(internal)','(external)','(unfolded)', '()']
for x in expressions:
    df['Screen Size'] = df['Screen Size'].str.replace(x, '', regex=True)

df['Screen Size'] = df['Screen Size'].apply(lambda x: x.split(' ')[0]).astype(float)

# Add the units of measurement into the relevant columns
df.rename(columns={'Mobile Weight':'Mobile Weight (grams)', 
                   'Front Camera':'Front Camera (MP)',
                   'RAM':'RAM (GB)',
                   'Back Camera': 'BackCamera (MP)',
                  'Battery Capacity':'Battery Capacity (mAh)',
                  'Screen Size':'Screen Size (inches)'}, inplace=True)

# Plot chart for the RAM Distribution
sns.set_style("darkgrid")
sns.set_palette('pastel')
plt.figure(figsize=(15, 5))
sns.countplot(x='RAM (GB)', data=df, edgecolor='black')
plt.xlabel('RAM (GB)')
plt.ylabel('Distribution')
plt.title('RAM Distribution')
plt.xticks(rotation=0);
plt.show()

# Plot the market share of the mobile companies
plt.figure(figsize=(15,5))
sns.barplot(x='Company Name', y='Launched Price (USA)', data=df, edgecolor='black')

plt.xticks(rotation=45)
plt.xlabel('Mobile Companies')
plt.ylabel('Market Share')
plt.title('Mobile Companies Market Share')

plt.show()

# Plot the number of phone brands per Company
plt.figure(figsize=(15, 5))

sns.countplot(x='Company Name', data=df, order=df['Company Name'].value_counts().index, edgecolor='black')

plt.xticks(rotation=45)
plt.xlabel('Mobile Companies')
plt.ylabel('Frequency')
plt.title('Distribution of Mobile Companies')

plt.show()