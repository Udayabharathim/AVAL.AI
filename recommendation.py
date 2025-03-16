import pandas as pd
import random
from tabulate import tabulate

# Paths to datasets (use relative paths or allow user input)
RECIPES_PATH = r"C:\Users\Elitebook 840 G6\Documents\AVAL2\data\exercise.csv - sheet1.csv"
EXERCISE_PATH = r"C:\Users\Elitebook 840 G6\Documents\AVAL2\data\recipes.csv - Sheet1.csv"

# Load datasets with error handling
try:
    recipes_df = pd.read_csv(RECIPES_PATH, encoding='latin-1')
    exercise_df = pd.read_csv(EXERCISE_PATH, encoding='latin-1')
except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure the dataset files exist at the specified paths.")
    exit(1)
except Exception as e:
    print(f"An unexpected error occurred while loading datasets: {e}")
    exit(1)

# Validate datasets
if recipes_df.empty or exercise_df.empty:
    print("Error: One or more datasets are empty. Please check the dataset files.")
    exit(1)

def validate_inputs(busy_morning, work_hours, exercise_freq, preferred_exercise_type, time_of_day_exercise, exercise_duration):
    """
    Validates user inputs to ensure they are within acceptable ranges.
    """
    if busy_morning not in ["yes", "no"]:
        raise ValueError("busy_morning must be 'yes' or 'no'.")
    if not (0 <= work_hours <= 24):
        raise ValueError("work_hours must be between 0 and 24.")
    if exercise_freq not in ["1-2 times", "3-4 times", "5-7 times"]:
        raise ValueError("exercise_freq must be one of '1-2 times', '3-4 times', or '5-7 times'.")
    if preferred_exercise_type not in ["low", "medium", "high"]:
        raise ValueError("preferred_exercise_type must be one of 'low', 'medium', or 'high'.")
    if time_of_day_exercise not in ["morning", "evening", "any"]:
        raise ValueError("time_of_day_exercise must be one of 'morning', 'evening', or 'any'.")
    if not (10 <= exercise_duration <= 120):
        raise ValueError("exercise_duration must be between 10 and 120 minutes.")

def generate_diet_and_exercise_plan(busy_morning, work_hours, exercise_freq, preferred_exercise_type, time_of_day_exercise, exercise_duration):
    """
    Generates a personalized weekly diet and exercise plan based on user inputs.
    Returns:
        diet_plan (str): HTML table for the diet plan.
        exercise_plan (str): HTML table for the exercise plan.
    """
    days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    meals = ["Breakfast", "Lunch", "Dinner"]

    diet_table_data = [["Day", "Breakfast", "Lunch", "Dinner"]]
    exercise_table_data = [["Day", "Exercise"]]

    recipe_name_column = recipes_df.columns[0]
    exercise_name_column = exercise_df.columns[0]

    # Filter exercises based on user preferences
    if "Intensity" in exercise_df.columns and "Duration" in exercise_df.columns:
        filtered_exercises = exercise_df[
            (exercise_df["Intensity"] == preferred_exercise_type) &
            (exercise_df["Duration"] <= exercise_duration)
        ]
    else:
        print("Warning: 'Intensity' or 'Duration' column not found in exercise dataset. Using all exercises.")
        filtered_exercises = exercise_df

    if filtered_exercises.empty:
        print("Warning: No exercises match the selected preferences. Using all exercises.")
        filtered_exercises = exercise_df

    # Determine exercise days based on frequency
    exercise_days = []
    if exercise_freq == "1-2 times":
        exercise_days = random.sample(days, k=2)
    elif exercise_freq == "3-4 times":
        exercise_days = random.sample(days, k=4)
    elif exercise_freq == "5-7 times":
        exercise_days = days

    for day in days:
        # Generate diet plan
        diet_row = [day]
        for meal in meals:
            recipe = recipes_df.sample(1)
            diet_row.append(recipe[recipe_name_column].values[0])
        diet_table_data.append(diet_row)

        # Generate exercise plan
        exercise_row = [day]
        if day in exercise_days:
            exercise = filtered_exercises.sample(1)
            exercise_row.append(exercise[exercise_name_column].values[0])
        else:
            exercise_row.append("Rest day")
        exercise_table_data.append(exercise_row)

    # Generate HTML tables
    diet_plan = tabulate(diet_table_data, headers="firstrow", tablefmt="html")
    exercise_plan = tabulate(exercise_table_data, headers="firstrow", tablefmt="html")

    return diet_plan, exercise_plan

def pcos_recommendation_chatbot(user_input, form_data):
    """
    Handles user input and generates personalized recommendations.
    Args:
        user_input (str): User's input (e.g., "get diet and exercise recommendations").
        form_data (dict): Dictionary containing user's lifestyle preferences.
    Returns:
        diet_plan (str): HTML table for the diet plan.
        exercise_plan (str): HTML table for the exercise plan.
    """
    if "recommendation" in user_input.lower() or "recommend" in user_input.lower():
        # Extract and validate form data
        busy_morning = form_data.get("busy_morning", "no")
        work_hours = int(form_data.get("work_hours", 8))
        exercise_freq = form_data.get("exercise_freq", "1-2 times")
        preferred_exercise_type = form_data.get("preferred_exercise_type", "low")
        time_of_day_exercise = form_data.get("time_of_day_exercise", "evening")
        exercise_duration = int(form_data.get("exercise_duration", 30))

        # Generate diet and exercise plans
        diet_plan, exercise_plan = generate_diet_and_exercise_plan(
            busy_morning, work_hours, exercise_freq, preferred_exercise_type, time_of_day_exercise, exercise_duration
        )
        return diet_plan, exercise_plan
    else:
        return None, None  # Return two values even if no recommendations are generated

# Example usage
if __name__ == "__main__":
    user_input = "get diet and exercise recommendations"
    form_data = {
        "busy_morning": "yes",
        "work_hours": 8,
        "exercise_freq": "3-4 times",
        "preferred_exercise_type": "medium",
        "time_of_day_exercise": "evening",
        "exercise_duration": 45
    }

    result = pcos_recommendation_chatbot(user_input, form_data)
    if isinstance(result, tuple):
        diet_plan, exercise_plan = result
        print("Diet Plan:\n", diet_plan)
        print("Exercise Plan:\n", exercise_plan)
    else:
        print(result)