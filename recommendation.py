from flask import Flask, request, jsonify
import pandas as pd
import random

app = Flask(__name__)

# Paths to CSV files (update these paths as needed)
recipes_path = r"C:\Users\Elitebook 840 G6\Documents\AVAL2\data\recipes.csv - Sheet1.csv"
exercise_path = r"C:\Users\Elitebook 840 G6\Documents\AVAL2\data\exercise.csv - sheet1.csv"

# Load datasets
recipes_df = pd.read_csv(recipes_path, encoding='latin-1')
exercise_df = pd.read_csv(exercise_path, encoding='latin-1')

def filter_recipes_based_on_time(recipes_df, max_time):
    return recipes_df[recipes_df['COOKING TIME(IN MINUTES)'] <= max_time]

def filter_exercises_based_on_time(exercise_df, max_time):
    return exercise_df[exercise_df['DURATION'] <= max_time]

def filter_exercises_based_on_time_of_day(exercise_df, time_of_day):
    return exercise_df[exercise_df['TIME OF DAY'].str.lower() == time_of_day.lower()]

def generate_diet_and_exercise_plan(busy_morning, work_hours, exercise_freq, preferred_exercise_type, time_of_day_exercise, exercise_duration):
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    meals = ["Breakfast", "Lunch", "Dinner"]

    diet_table_data = [["Day", "Breakfast", "Lunch", "Dinner"]]
    exercise_table_data = [["Day", "Exercise", "Duration"]]

    if work_hours > 8:
        recipes_df_filtered = filter_recipes_based_on_time(recipes_df, 30)
    else:
        recipes_df_filtered = recipes_df

    if busy_morning == "yes":
        breakfast_recipes = filter_recipes_based_on_time(recipes_df_filtered, 15)
    else:
        breakfast_recipes = recipes_df_filtered

    exercise_df_filtered = filter_exercises_based_on_time(exercise_df, exercise_duration)

    for day in days:
        diet_row = [day]
        for meal in meals:
            if meal == "Breakfast" and busy_morning == "yes":
                recipe = breakfast_recipes.sample(1)
            else:
                recipe = recipes_df_filtered.sample(1)
            diet_row.append(recipe.iloc[0]['RECIPE NAME'])
        diet_table_data.append(diet_row)

    for day in days:
        exercise_row = [day]
        if time_of_day_exercise == "morning":
            exercise = filter_exercises_based_on_time_of_day(exercise_df_filtered, "morning").sample(1)
        elif time_of_day_exercise == "evening":
            exercise = filter_exercises_based_on_time_of_day(exercise_df_filtered, "evening").sample(1)
        elif time_of_day_exercise == "both":
            exercise = filter_exercises_based_on_time_of_day(exercise_df_filtered, "both").sample(1)
        else:
            exercise = exercise_df_filtered.sample(1)

        exercise_row.extend([
            exercise.iloc[0]['EXERCISE NAME'],
            f"{exercise.iloc[0]['DURATION']} mins"
        ])
        exercise_table_data.append(exercise_row)

    return diet_table_data, exercise_table_data

@app.route('/generate-plan', methods=['POST'])
def generate_plan():
    data = request.get_json()

    busy_morning = data.get('busy_morning')
    work_hours = int(data.get('work_hours'))
    exercise_freq = data.get('exercise_freq')
    preferred_exercise_type = data.get('preferred_exercise_type')
    time_of_day_exercise = data.get('time_of_day_exercise')
    exercise_duration = int(data.get('exercise_duration'))

    diet_table_data, exercise_table_data = generate_diet_and_exercise_plan(
        busy_morning, work_hours, exercise_freq, preferred_exercise_type, time_of_day_exercise, exercise_duration
    )

    return jsonify({
        'diet_table_data': diet_table_data,
        'exercise_table_data': exercise_table_data
    })

@app.route('/')
def index():
    return "Welcome to the PCOS Lifestyle Recommendation System!"

if __name__ == '__main__':
    app.run(debug=True)