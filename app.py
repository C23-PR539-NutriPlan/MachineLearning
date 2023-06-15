import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
# import _mysql_connector as mysql
import mysql.connector as mysql
import os

import tensorflow as tf
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('Recommender_Model.h5')

# Load the data and other necessary variables
data = pd.read_csv('new_recipe.csv')
df = data[['Name', 'RecipeCategory', 'RecipeIngredientParts',
           'Calories','FatContent','SaturatedFatContent','CholesterolContent',
           'SodiumContent','CarbohydrateContent',
           'FiberContent', 'SugarContent', 'ProteinContent']].copy()
ingredient_encoding = np.load('ingredient_encoding.npy', allow_pickle=True)
calories_normalized = np.load('calories_normalized.npy', allow_pickle=True)


def create_db_connection():
    connection = mysql.connect(
        host='34.101.224.88',
        user='root',
        password='pass',
        database='NutriPlan_db'
    )
    return connection


@app.route('/predict', methods=['POST'])
def predict():

    connection = create_db_connection()
    cursor = connection.cursor()

    inp = request.get_json(force=True)
    print(inp)

    # # Define the user input
    # user_calories = 700
    # user_allergies = ['milk', 'eggs']
    # user_favorites = ['pizza', 'chocolate']

    # # Prepare the input data
    # input_data = {
    #     'user_calories': user_calories,
    #     'user_allergies': user_allergies,
    #     'user_favorites': user_favorites
    # }

    # Process the input data and make predictions
    user_input = np.zeros((1, 9353))
    # user_input[0, 0] = user_calories
    user_input[0, 0] = inp['user_calories']

    # Get the normalized calorie prediction from the model
    user_calories_normalized = model.predict(user_input)

    # Filter the dataset based on the user's allergic ingredients and favorite food
    filtered_indices = []
    for idx, row in df.iterrows():
        if all(allergy.lower() not in row['RecipeIngredientParts'].lower() for allergy in inp['user_allergies']) and any(favorite.lower() in row['Name'].lower() for favorite in inp['user_favorites']):
            filtered_indices.append(idx)

    filtered_indices = np.array(filtered_indices)
    valid_indices = np.where(filtered_indices < df.shape[0])[0]

    # Find food items with similar calorie values
    food_indices = np.argsort(np.abs(calories_normalized - user_calories_normalized), axis=None)

    # Ensure the food_indices array is within the valid range
    food_indices = food_indices[food_indices < len(filtered_indices)]
    food_indices = food_indices[:30]  # Adjust the number of recommendations as needed

    recommendations = [df.iloc[filtered_indices[valid_indices[idx]], :] for idx in food_indices]

    # Store the recommended food items in an array
    # Set the desired calorie range
    max_calories = inp['user_calories']  # Maximum calorie value
    recommended_items = []
    for item in recommendations:
        if len(recommended_items) == 30:
            break
        # if item['Images'] == "character(0)":
        #     continue
        # else:
            # # Remove unnecessary characters and split the URLs
            # urls = item['Images'].replace('c(', '').replace(')', '').replace('\n', '').replace('"', '').split(',')
            # # Remove empty elements and leading/trailing spaces
            # urls = [url.strip() for url in urls if url.strip()]
            
            # item['Images'] = urls[0]
        calories = item['Calories']
        if calories <= max_calories:
            recommended_items.append(item.to_dict())  # Add the values of the item to the array
    

    for item in recommended_items:
        cursor.execute("SELECT id FROM foods WHERE name=%s", (item['Name'],))
        out = cursor.fetchall()
        if len(out) == 0:
            name = item['Name']
            recipe_category = item['RecipeCategory']
            calories = item['Calories']
            fat_content = item['FatContent']
            saturated_fat_content = item['SaturatedFatContent']
            cholesterol_content = item['CholesterolContent']
            sodium_content = item['SodiumContent']
            carbohydrate_content = item['CarbohydrateContent']
            fiber_content = item['FiberContent']
            sugar_content = item['SugarContent']
            protein_content = item['ProteinContent']

            query = 'INSERT INTO foods (name, recipeCategory, calories, fatContent, saturatedFatContent, cholesterolContent, sodiumContent, carbohydrateContent, fiberContent, sugarContent, proteinContent) VALUES ("{}", "{}", {}, {}, {}, {}, {}, {}, {}, {}, {});'.format(name, recipe_category, calories, fat_content, saturated_fat_content, cholesterol_content, sodium_content, carbohydrate_content, fiber_content, sugar_content, protein_content)
            
            cursor.execute(query)
            connection.commit()
    
        
    cursor.execute("DELETE FROM recommendation WHERE userID=%s", (inp['user_id'],))
    connection.commit()
    for item in recommended_items:
        cursor.execute("SELECT id FROM foods WHERE name=%s", (item['Name'],))
        out = cursor.fetchall()
        foodid = out[0][0]
        query = "INSERT INTO recommendation (userID, foodID) VALUES (%s, %s)"
        params = (inp['user_id'], foodid)
        cursor.execute(query, params)
        connection.commit()

            

    # Print the recommended food items
    # print("Recommended Food Items:")
    # for item in recommended_items:
    #     print(item)
    
   
    return jsonify(recommended_items)

if __name__ == "__main__":
    port = int(os.getenv("PORT"))
    app.run(host='0.0.0.0', port=port)
    # app.run(debug=True)
