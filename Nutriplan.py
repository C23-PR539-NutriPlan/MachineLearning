#!/usr/bin/env python
# coding: utf-8

# # Getting the Data

# In[2]:


# get_ipython().system('pip install pandas')
# get_ipython().system('pip install sklearn')


# In[3]:


# get_ipython().system('pip install tensorflow')
# # !pip install --upgrade tensorflow-gpu --user


# In[4]:


# get_ipython().system('pip install matplotlib')


# In[5]:


import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle


# In[30]:


# data=pd.read_csv('recipe.csv', error_bad_lines=False, engine="python")
data=pd.read_csv('new_recipe.csv')


# In[31]:


total_rows, total_attributes = data.shape
print('Jumlah data:', total_rows)
print("Jumlah atribut:", total_attributes)
data.head()


# In[32]:


# Melihat kolom-kolom pada dataset
data.info()


# In[9]:


# df = data[['RecipeId', 'RecipeCategory', 'Keywords', 'RecipeIngredientParts']]
# df


# In[10]:


# df['RecipeCategory'].nunique()
# df['RecipeCategory'].unique()


# # Preprocessing

# In[ ]:





# In[33]:


df = data[['RecipeId','Name', 'Images', 'RecipeCategory','RecipeIngredientParts',
           'Calories','FatContent','SaturatedFatContent','CholesterolContent',
           'SodiumContent','CarbohydrateContent',
           'FiberContent', 'SugarContent', 'ProteinContent']].copy()


# In[12]:


df.head()


# Cek missing values

# In[34]:


df.isnull().sum()


# Menghapus (drop) baris yang mengandung missing values

# In[35]:


df = df.dropna()
df.isnull().sum()


# Cek Duplicate

# In[36]:


df.duplicated(keep=False).sum()


# In[37]:


fig, ax = plt.subplots(figsize=(10, 8))
plt.title('Frequency Histogram')
plt.ylabel('Frequency')
plt.xlabel('Calories')
ax.hist(df.Calories.to_numpy(),bins=[0,100,200,300,400,500,600,700,800,900,1000,1000,2000,3000,5000],linewidth=0.5, edgecolor="white")
plt.show()


# In[38]:


import pylab 
import scipy.stats as stats
stats.probplot(data.Calories.to_numpy(), dist="norm", plot=pylab)
pylab.show()


# In[39]:


#TO DO: ubah nilai max bergantung sama input user, sementara hard code
max_Calories=2000
max_daily_fat=100
max_daily_Saturatedfat=13
max_daily_Cholesterol=300
max_daily_Sodium=2300
max_daily_Carbohydrate=325
max_daily_Fiber=40
max_daily_Sugar=40
max_daily_Protein=200
max_list=[max_Calories,max_daily_fat,max_daily_Saturatedfat,max_daily_Cholesterol,max_daily_Sodium, max_daily_Carbohydrate, max_daily_Fiber, max_daily_Sugar, max_daily_Protein]


# In[42]:


# extracted_data=df.copy()
# extracted_data.columns[4:13]


# In[44]:


extracted_data=df.copy()
for column,maximum in zip(extracted_data.columns[5:13],max_list):
    extracted_data=extracted_data[extracted_data[column]<maximum]


# In[45]:


extracted_data.info()


# In[46]:


extracted_data.iloc[:,5:13].corr()


# In[47]:


# get_ipython().system('pip install scikit-learn')


# In[49]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
prep_data=scaler.fit_transform(extracted_data.iloc[:,5:13].to_numpy())


# In[50]:


prep_data


# # Build and Training Model

# In[51]:


from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix


# In[52]:


#load the dataset
data = extracted_data.copy()
data.head()


# In[53]:


data.info()


# In[54]:


print(data['Calories'].min())
print(data['Calories'].max())


# In[55]:

print("238")

# Preprocess the data
# Encode the ingredients using one-hot encoding
ingredients = data['RecipeIngredientParts'].str.split(',').fillna('')
mlb = MultiLabelBinarizer()
ingredient_encoding = csr_matrix(mlb.fit_transform(ingredients))
# ingredient_encoding.info

print("247")
# In[56]:


# Normalize the calorie column
scaler = StandardScaler()
calories_normalized = scaler.fit_transform(data[['Calories']])
print("254")

# In[57]:


# Specify the user's daily calorie needs, allergic ingredients, and favorite food
user_calories = 700  # Example: User's daily calorie needs (each meal)
user_allergies = ['milk', 'eggs']  # Example: User's allergic ingredients
user_favorites = ['pizza', 'chocolate']  # Example: User's favorite food
print("263")

# In[58]:


# Filter the dataset based on the user's allergic ingredients and favorite food
filtered_indices = []
for idx, row in data.iterrows():
    if all(allergy.lower() not in row['RecipeIngredientParts'].lower() for allergy in user_allergies) and any(
            favorite.lower() in row['Name'].lower() for favorite in user_favorites):
        filtered_indices.append(idx)
print("274")

# In[59]:


# Check if there are any filtered food items
if len(filtered_indices) == 0:
    print("No suitable food items found for the given preferences and allergies.")
    similar_items = 15  # Number of similar items to recommend
    similar_indices = np.argsort(np.abs(cal_train - user_calories_normalized))[:similar_items]
    recommendations = [data.iloc[valid_indices[idx], :] for idx in similar_indices]
     # Print the recommended food items
    print("Recommended Food Items:")
    recommended_names = set()
    for item in recommendations:
        item_tuple = tuple(item)
        if item_tuple not in recommended_names:
            print(item)
            recommended_names.add(item_tuple)
else:
    # Convert filtered_indices to numpy array and check for valid indices
    filtered_indices = np.array(filtered_indices)
    valid_indices = np.where(filtered_indices < data.shape[0])[0]

    # Check if there are any valid indices
    if len(valid_indices) == 0:
        print("No suitable food items found for the given preferences and allergies.")
    else:
        # Prepare the model input
        X = ingredient_encoding[filtered_indices[valid_indices]]

        # Normalize the calories for the filtered indices
        calories_normalized_filtered = calories_normalized[filtered_indices[valid_indices]]

        # Split the data into training and testing sets
        X_train, X_test, cal_train, cal_test = train_test_split(
            X, calories_normalized_filtered, test_size=0.2, random_state=42
        )

        # Build the model architecture
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(X_train, cal_train, epochs=10, batch_size=32)

        # # Recommend food based on user's preferences and daily calorie needs
        # user_input = np.zeros((1, X_train.shape[1]))  # Initialize the user input with zero ratings
        # user_input[0, 0] = user_calories  # Set the user's daily calorie needs
        # # user_input[0, 0] = 200  # Set the user's daily calorie needs

        user_calories_normalized = model.predict(user_input)

        # print(user_calories_normalized)

        # Find food items with similar calorie values
        food_indices = np.argsort(np.abs(cal_train - user_calories_normalized), axis=None)
        recommendations = [data.iloc[filtered_indices[valid_indices[idx]], :] for idx in food_indices]
#         print(len(recommendations))

#         # Set the desired calorie range
#         min_calories = 0  # Minimum calorie value
#         max_calories = user_calories_normalized # Maximum calorie value

#         # Find food items within the desired calorie range
#         filtered_recommendations = []
#         for item in recommendations:
#             calories = item['Calories']
#             if min_calories <= calories <= max_calories:
#                 filtered_recommendations.append(item)
#                 # 15 food only
#                 if len(filtered_recommendations) == 15:  # Limit the number of recommendations to 15
#                     break

#         # Print the recommended food items
#         print(len(filtered_recommendations))
#         print("Recommended Food Items:")
#         for item in filtered_recommendations:
#             print(item)


# # In[ ]:

print("363")
# model.save('Recommender_Model.h5')

##########################################
pickle.dump(model, open('model.pkl', 'wb'))

modelFinal = pickle.load(open('model.pkl', 'rb'))

user_calories = 700
user_allergies = ['milk', 'eggs']
user_favorites = ['pizza', 'chocolate']

# Prepare the input data
input_data = {
    'user_calories': user_calories,
    'user_allergies': user_allergies,
    'user_favorites': user_favorites
}
# Recommend food based on user's preferences and daily calorie needs
user_input = np.zeros((1, X_train.shape[1]))  # Initialize the user input with zero ratings
user_input[0, 0] = user_calories  # Set the user's daily calorie needs
# user_input[0, 0] = 200  # Set the user's daily calorie needs

# predictions = model.predict(input_data)
predictions = modelFinal.predict(user_input)
print("huhuhu")
print(predictions)



# Process the predictions and return the recommended food items to the user
# You can customize this part based on
# %%
