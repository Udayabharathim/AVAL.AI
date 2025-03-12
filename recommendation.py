import pandas as pd
import random
from tabulate import tabulate

# Paths to datasets
recipes_path = r"C:\Users\Elitebook 840 G6\Documents\AVAL2\data\recipes.csv - Sheet1.csv"
exercise_path = r"C:\Users\Elitebook 840 G6\Documents\AVAL2\data\exercise.csv - sheet1.csv"

# Load datasets
recipes_df = pd.read_csv(recipes_path, encoding='latin-1')
exercise_df = pd.read_csv(exercise_path, encoding='latin-1')

def generate_diet_and_exercise_plan():
    """
    Generates a personalized weekly diet and exercise plan.
    Returns:
        diet_plan (str): HTML table for the diet plan.
        exercise_plan (str): HTML table for the exercise plan.
    """
    days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    meals = ["Breakfast", "Lunch", "Dinner"]

    diet_table_data = [["Days", "Breakfast", "Lunch", "Dinner"]]
    exercise_table_data = [["Days", "Morning Exercise", "Evening Exercise"]]

    recipe_name_column = recipes_df.columns[0]
    exercise_name_column = exercise_df.columns[0]

    for day in days:
        diet_row = [day]
        for meal in meals:
            recipe = recipes_df.sample(1)
            diet_row.append(recipe[recipe_name_column].values[0])
        diet_table_data.append(diet_row)

        exercise_row = [day]
        morning_exercise = exercise_df.sample(1)
        evening_exercise = exercise_df.sample(1)
        exercise_row.extend([morning_exercise[exercise_name_column].values[0], evening_exercise[exercise_name_column].values[0]])
        exercise_table_data.append(exercise_row)

    diet_plan = tabulate(diet_table_data, headers="firstrow", tablefmt="html")
    exercise_plan = tabulate(exercise_table_data, headers="firstrow", tablefmt="html")

    return diet_plan, exercise_plan

def pcos_recommendation_chatbot(user_input):
    """
    Handles user input and generates personalized recommendations.
    Args:
        user_input (str): User's input (e.g., "get diet and exercise recommendations").
    Returns:
        diet_plan (str): HTML table for the diet plan.
        exercise_plan (str): HTML table for the exercise plan.
    """
    if "recommendation" in user_input or "recommend" in user_input:
        print("\nLet's collect more information for a personalized lifestyle plan.")
        busy_morning = input("Do you usually have a busy morning schedule? (yes/no): ").strip().lower()
        work_hours = int(input("On average, how many hours do you spend at work or school each day? (0-12): ").strip())
        exercise_freq = input("How often do you exercise each week? (never/1-2 times/3-5 times/daily): ").strip().lower()
        preferred_exercise_type = input("Do you prefer high-intensity or low-intensity exercises? (high/low): ").strip().lower()
        time_of_day_exercise = input("When do you prefer to exercise? (morning/evening/both): ").strip().lower()
        exercise_duration = int(input("How much time can you dedicate to each exercise session? (in minutes): ").strip())

        diet_plan, exercise_plan = generate_diet_and_exercise_plan()
        return diet_plan, exercise_plan
    else:
        return None, None
    