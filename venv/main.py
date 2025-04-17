import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

# Load the dataset
df=pd.read_csv("/Users/sujalkarki/Desktop/Mobile Project /Mobiles.csv",encoding='latin1')
print(df.head(5)) #display the first 5 rows
print(df.columns)  #display the columns of the dataset
print(df.isnull().sum)
print(df.info())   # Get the summary of the dataset

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

#__________ Add the units of measurement into the relevant columns_________
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
plt.xticks(rotation=0)
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

# _______________Top 10 Phones with Largest Battery____________
top_battery = df.sort_values(by='Battery Capacity (mAh)', ascending=False).head(10)

# plot the top 10 phone with largest battery 
plt.figure(figsize=(12,6))
sns.barplot(x='Model Name', y='Battery Capacity (mAh)', data=top_battery, palette='crest')
plt.title('Top 10 Phones with Largest Battery')
plt.xticks(rotation=45)
plt.show()

#_____________ Most Expensive Phone per Company________________
idx = df.groupby('Company Name')['Launched Price (USA)'].idxmax()
most_expensive = df.loc[idx]

# plot the most expensive phone per company
plt.figure(figsize=(12,6))
sns.barplot(x='Company Name', y='Launched Price (USA)', data=most_expensive, palette='magma')
plt.title('Most Expensive Phone per Company')
plt.xticks(rotation=45)
plt.show()

# _____________Correlation Analysis______________
plt.figure(figsize=(12,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# ____________Analysis for Best Camera phone _____________
df['Total Camera (MP)'] = df['Front Camera (MP)'] + df['BackCamera (MP)']
top_camera_phones = df.sort_values(by='Total Camera (MP)', ascending=False).head(10)
print(top_camera_phones[['Model Name', 'Company Name', 'Front Camera (MP)', 'BackCamera (MP)', 'Total Camera (MP)']])

plt.figure(figsize=(12,6))
sns.barplot(x='Model Name', y='Total Camera (MP)', data=top_camera_phones, hue='Company Name', dodge=False, palette='viridis')
plt.title('Top 10 Phones with Best Camera Setup (Front + Back MP)')
plt.ylabel('Total Camera Megapixels')
plt.xlabel('Phone Model')
plt.xticks(rotation=45)
plt.legend(title='Brand')
plt.tight_layout()
plt.show()


# _____________Analysis for Best RAM mobile Phone in the range of 20000.____________
# Filter for phones launched in India under ₹20,000.
phones_under_20k = df[df['Launched Price (India)'] <= 20000]
# Sort them by RAM
best_ram_under_20k = phones_under_20k.sort_values(by='RAM (GB)', ascending=False)
# Display top 10
print(best_ram_under_20k[['Model Name', 'Company Name', 'RAM (GB)', 'Launched Price (India)']].head(10))

# Visualize the top 10 phones with the best RAM under 20000
plt.figure(figsize=(12,6))
sns.barplot(
    x='Model Name',
    y='RAM (GB)',
    data=best_ram_under_20k.head(10),
    hue='Company Name',
    dodge=False,
    palette='pastel'
)

plt.title('Top RAM Phones Under ₹20,000')
plt.xlabel('Model Name')
plt.ylabel('RAM (GB)')
plt.xticks(rotation=45)
plt.legend(title='Brand')
plt.tight_layout()
plt.show()

# ___________Analysis 
df['Price Diff (USA - India)'] = df['Launched Price (USA)'] - df['Launched Price (India)']
df['Price Diff (China - India)'] = df['Launched Price (China)'] - df['Launched Price (India)']
df['Price Diff (Dubai - India)'] = df['Launched Price (Dubai)'] - df['Launched Price (India)']
df['Price Diff (Pakistan - India)'] = df['Launched Price (Pakistan)'] - df['Launched Price (India)']

# Find Phones with the Highest Price Gaps
price_diff_cols = [
    'Price Diff (USA - India)',
    'Price Diff (China - India)',
    'Price Diff (Dubai - India)',
    'Price Diff (Pakistan - India)'
]
top_diff = df[['Model Name', 'Company Name'] + price_diff_cols].sort_values(by='Price Diff (USA - India)', ascending=False).head(10)
print(top_diff)
top_diff_chart = df.sort_values(by='Price Diff (USA - India)', ascending=False).head(10)

# Visualize Price Difference (USA vs India)
plt.figure(figsize=(12,6))
sns.barplot(x='Model Name', y='Price Diff (USA - India)', data=top_diff_chart, hue='Company Name', dodge=False, palette='magma')
plt.title('Top 10 Phones with Highest Price Difference (USA vs India)')
plt.ylabel('Price Difference (₹)')
plt.xlabel('Model Name')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# _______________Analysis for  top 10 phone by Overall Score_________________
# Total Camera Score (Front + Back)
df['Total Camera (MP)'] = df['Front Camera (MP)'] + df['BackCamera (MP)']

# Manual Min-Max Normalization
def min_max_normalize(series):
    return (series - series.min()) / (series.max() - series.min())

# Normalize each feature manually
df['RAM_norm'] = min_max_normalize(df['RAM (GB)'])
df['Battery_norm'] = min_max_normalize(df['Battery Capacity (mAh)'])
df['Camera_norm'] = min_max_normalize(df['Total Camera (MP)'])
df['Screen_norm'] = min_max_normalize(df['Screen Size (inches)'])
df['Price_norm'] = min_max_normalize(df['Launched Price (India)'])

# Calculate overall score (higher is better, lower price is better)
df['Overall Score'] = (
    df['RAM_norm'] +
    df['Battery_norm'] +
    df['Camera_norm'] +
    df['Screen_norm'] -
    df['Price_norm']  # Subtract price because higher price is less desirable
)

# Get the best phone model
best_phone = df.sort_values(by='Overall Score', ascending=False).head(1)

# Display the best phone
print("📱 Best Phone Overall:")
print(best_phone[['Model Name', 'Company Name', 'Overall Score']])

# Top 10 phones by Overall Score
top_10_overall = df.sort_values(by='Overall Score', ascending=False).head(10)

# Top 10 phones by Overall Score
top_10_overall = df.sort_values(by='Overall Score', ascending=False).head(10)

# Visualize the top 10 phone by Overall Score
plt.figure(figsize=(12, 6))
sns.barplot(
    x='Model Name',
    y='Overall Score',
    hue='Company Name',
    data=top_10_overall,
    dodge=False,
    palette='rocket'
)
plt.title('Top 10 Best Overall Phones (Based on Combined Specs)', fontsize=14)
plt.xlabel('Phone Model')
plt.ylabel('Overall Score')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Brand')
plt.tight_layout()
plt.show()

# ____________Analysis for Best Country to Buy a Phone_________
# _______List of countries and their launch price columns________
country_price_columns = {
    'India': 'Launched Price (India)',
    'USA': 'Launched Price (USA)',
    'China': 'Launched Price (China)',
    'Dubai': 'Launched Price (Dubai)',
    'Pakistan': 'Launched Price (Pakistan)'
}

# Calculate average price per country
avg_prices = {country: df[col].mean() for country, col in country_price_columns.items()}
avg_prices_df = pd.DataFrame(list(avg_prices.items()), columns=['Country', 'Average Price'])


















