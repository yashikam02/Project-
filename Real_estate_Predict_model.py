#!/usr/bin/env python
# coding: utf-8

# DATA CLEANING AND PREPROCESSING

# In[1]:


import pandas as pd 
import numpy as np
df=pd.read_csv(r"Real_Estate.csv")
df


# In[2]:


#to CHeck if there are any null values present in the Dataset
df.isnull().sum()


# There are no null values in the dataset.

# In[3]:


df.duplicated().sum()


# In[4]:


x=df.iloc[:,:-1]
y=df.iloc[:,-1]
x,y


# Exploratory data analysis

# In[5]:


descriptive_stats = df.describe()

print(descriptive_stats)


# DATA VISUALIZATION

# In[6]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

# Creating  histograms
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12))
fig.suptitle('Histograms of Real Estate Data', fontsize=16)

cols = ['House age', 'Distance to the nearest MRT station', 'Number of convenience stores',
        'Latitude', 'Longitude', 'House price of unit area']

for i, col in enumerate(cols):
    sns.histplot(df[col], kde=True, ax=axes[i//2, i%2])
    axes[i//2, i%2].set_title(col)
    axes[i//2, i%2].set_xlabel('')
    axes[i//2, i%2].set_ylabel('')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# OUTCOMES:
# 
# 1)House Age: The distribution of house ages is relatively uniform, with a slight increase in the number of newer properties, indicating that a significant portion of the properties are relatively young:
# 
# 2)Distance to the Nearest MRT Station: The majority of properties are located in close proximity to an MRT station, as evidenced by the high frequency of shorter distances. However, there is a long tail extending towards higher distances, which suggests the presence of some properties that are located quite far from MRT stations.
# 
# 3)Number of Convenience Stores: The data reveals a wide range of values for the number of convenience stores, with distinct peaks at certain counts such as 0, 5, and 10. This indicates that there are certain common configurations in the availability of convenience stores across properties.
# 
# 4)Latitude and Longitude: Both latitude and longitude distributions are relatively concentrated, implying that the properties are located within a geographically confined area, suggesting a focused region for the properties under analysis.
# 
# 5)House Price of Unit Area: The distribution of house prices per unit area is right-skewed, with a higher concentration of properties in the lower price range. As the price increases, the number of properties decreases, indicating fewer high-priced properties relative to the majority of lower-priced ones.

# In[7]:


#exploring the relationships between these variables and the house price using Scatter Plot
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
fig.suptitle('Scatter Plots with House Price of Unit Area', fontsize=16)

# Scatter plot for each variable against the house price
sns.scatterplot(data=df, x='House age', y='House price of unit area', ax=axes[0, 0])
sns.scatterplot(data=df, x='Distance to the nearest MRT station', y='House price of unit area', ax=axes[0, 1])
sns.scatterplot(data=df, x='Number of convenience stores', y='House price of unit area', ax=axes[1, 0])
sns.scatterplot(data=df, x='Latitude', y='House price of unit area', ax=axes[1, 1])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# The scatter plots revealed the following key relationships between factors and house prices:
# 
# 1.House Age vs. House Price: No strong linear relationship, though both very new and very old houses may have higher prices.
# 
# 2.Distance to the Nearest MRT Station vs. House Price: A clear negative correlation, with house prices decreasing as the distance to the MRT station increases.
# 
# 3.Number of Convenience Stores vs. House Price: A positive relationship, where houses near more convenience stores tend to have higher prices.
# 
# 4.Latitude vs. House Price: A weak pattern suggesting that certain latitudes may correspond to higher or lower house prices, potentially reflecting neighborhood desirability.

# In[8]:


#correlation Matrix
df['Transaction date'] = pd.to_datetime(df['Transaction date'], errors='coerce')

# Convert the datetime to a Unix timestamp (seconds since epoch)
df['transaction_date_numeric'] = df['Transaction date'].astype('int64') // 10**9  # Convert to seconds

# Now, you can use the numeric date for correlation analysis
numeric_df = df.select_dtypes(include='number')
correlation_matrix = numeric_df.corr()

# Plotting the correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()


# Overall, the most significant factors affecting house prices in this dataset are proximity to MRT stations and the number of nearby convenience stores. Geographical location (latitude and longitude) and house age appear to have a lesser impact on the price.

# MODEL BUILDING

# 1)Splitting the dataset

# In[9]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Selecting features and target variable
features = ['Distance to the nearest MRT station', 'Number of convenience stores', 'Latitude', 'Longitude']
target = 'House price of unit area'

X = df[features]
y = df[target]

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 2)Model Training

# In[10]:


# Model initialization
model = LinearRegression()

# Training the model
model.fit(X_train, y_train)


# 3)Evaluation and Visualization

# In[11]:


# Making predictions using the Linear regression
y_pred = model.predict(X_test)
y_pred


# In[12]:


y_test


# In[13]:


model.score(X_test,y_test)


# In[14]:


# Visualization: Actual vs. Predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted House Prices (Gradient Boosting Regressor)')
plt.show()

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")


# In[15]:


import pickle
pickle.dump(model,open('model.pkl','wb'))


# In[16]:


import dash
from dash import html, dcc, Input, Output, State
import pandas as pd

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.Div([
        html.H1("Real Estate Price Prediction", style={'text-align': 'center'}),
        
        html.Div([
            dcc.Input(id='distance_to_mrt', type='number', placeholder='Distance to MRT Station (meters)',
                      style={'margin': '10px', 'padding': '10px'}),
            dcc.Input(id='num_convenience_stores', type='number', placeholder='Number of Convenience Stores',
                      style={'margin': '10px', 'padding': '10px'}),
            dcc.Input(id='latitude', type='number', placeholder='Latitude',
                      style={'margin': '10px', 'padding': '10px'}),
            dcc.Input(id='longitude', type='number', placeholder='Longitude',
                      style={'margin': '10px', 'padding': '10px'}),
            html.Button('Predict Price', id='predict_button', n_clicks=0,
                        style={'margin': '10px', 'padding': '10px', 'background-color': '#007BFF', 'color': 'white'}),
        ], style={'text-align': 'center'}),
        
        html.Div(id='prediction_output', style={'text-align': 'center', 'font-size': '20px', 'margin-top': '20px'})
    ], style={'width': '50%', 'margin': '0 auto', 'border': '2px solid #007BFF', 'padding': '20px', 'border-radius': '10px'})
])

# Define callback to update output
@app.callback(
    Output('prediction_output', 'children'),
    [Input('predict_button', 'n_clicks')],
    [State('distance_to_mrt', 'value'), 
     State('num_convenience_stores', 'value'),
     State('latitude', 'value'),
     State('longitude', 'value')]
)
def update_output(n_clicks, distance_to_mrt, num_convenience_stores, latitude, longitude):
    if n_clicks > 0 and all(v is not None for v in [distance_to_mrt, num_convenience_stores, latitude, longitude]):
        # Prepare the feature vector
        features = pd.DataFrame([[distance_to_mrt, num_convenience_stores, latitude, longitude]], 
                                columns=['Distance to the nearest MRT station', 'Number of convenience stores', 'Latitude', 'Longitude'])
        # Predict
        prediction = model.predict(features)[0]
        return f'Predicted House Price of Unit Area: {prediction:.2f}'
    elif n_clicks > 0:
        return 'Please enter all values to get a prediction'
    return ''

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)


# In[ ]:




