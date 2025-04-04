import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
#Load dataset from CSV file 
data = pd.read_csv("/content/traffic_data.csv") 
#Convert Date column to datetime format data['Date'] = pd.to_datetime(data['Date']) 
# Display basic information 
print(data.info()) 
print(data.describe()) 
#Visualize traffic trends over time 
plt.figure(figsize=(12, 6)) 
sns.lineplot(x=data['Date'], y=data['PageViews'], marker='0') 
plt.title('Website Traffic Over Time') 
plt.xlabel('Date') 
plt.ylabel('Page Views') 
plt.xticks (rotation=45) 
plt.show() 
# Analyze bounce rates over time 
plt.figure(figsize=(12, 6)) 
sns.lineplot(x=data['Date'], y=data['Bounce Rate'], marker='o', color='r') 
plt.title('Bounce Rate Over Time') 
plt.xlabel('Date') 
plt.ylabel('Bounce Rate (%)') 
plt.xticks (rotation=45) 
plt.show() 
# Bar chart for Unique Visitors per day 
plt.figure(figsize=(12, 6)) 
sns.barplot(x=data['Date'], y=data['UniqueVisitors'], palette='viridis') 
plt.title('Unique Visitors Per Day') 
plt.xlabel('Date') 
plt.ylabel("Unique Visitors') 
plt.xticks(rotation=45) 
plt.show()