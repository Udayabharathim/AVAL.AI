from flask import Flask, render_template, request, jsonify
import pandas as pd
import random
import numpy as np

app = Flask(__name__)

# Load datasets
recipes_df = pd.read_csv(r"C:\Users\Elitebook 840 G6\Documents\AVAL2\data\recipes.csv - Sheet1.csv", encoding='latin-1')
exercise_df = pd.read_csv(r"C:\Users\Elitebook 840 G6\Documents\AVAL2\data\exercise.csv - sheet1.csv", encoding='latin-1')
weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

def generate_plan(user_input):
    """Generates a personalized diet and exercise plan based on user inputs."""
    diet_data = recipes_df.copy()
    exercise_data = exercise_df.copy()

    # Filter meals based on user constraints
    if "COOKING TIME(IN MINUTES)" in diet_data.columns:  
        if user_input["busy_morning"] == "yes":
            diet_data = diet_data[diet_data["COOKING TIME(IN MINUTES)"] <= 15]  # Quick breakfast
        if user_input["work_hours"] > 8:
            diet_data = diet_data[diet_data["COOKING TIME(IN MINUTES)"] <= 30]  # Quick meals
    
    diet_plan = []
    for _ in range(7):
        diet_plan.append({
            "Breakfast": diet_data.sample(1).to_dict(orient='records')[0] if not diet_data.empty else "N/A",
            "Lunch": diet_data.sample(1).to_dict(orient='records')[0] if not diet_data.empty else "N/A",
            "Dinner": diet_data.sample(1).to_dict(orient='records')[0] if not diet_data.empty else "N/A",
        })
    
    exercise_plan = []
    for i in range(7):
        day_exercise = {"Day": weekdays[i]}
        available_exercises = exercise_data.sample(frac=1).reset_index(drop=True)

        if user_input["time_of_day_exercise"] in ["morning", "both"]:
            day_exercise["Morning"] = available_exercises.sample(1).to_dict(orient='records')[0] if not available_exercises.empty else "N/A"
        if user_input["time_of_day_exercise"] in ["evening", "both"]:
            day_exercise["Evening"] = available_exercises.sample(1).to_dict(orient='records')[0] if not available_exercises.empty else "N/A"
        
        exercise_plan.append(day_exercise)
    print(diet_plan)
    print("\n")
    print(exercise_plan)
    return diet_plan, exercise_plan

@app.route('/')
def home():
    return render_template('recommendation.html')

@app.route('/get_recommendation', methods=['POST'])
def get_recommendation():
    user_input = request.json
    diet_plan, exercise_plan = generate_plan(user_input)

    # Remove NaN values
    def clean_data(plan):
        cleaned_plan = []
        for day in plan:
            cleaned_day = {}
            for meal, details in day.items():
                if isinstance(details, dict):  # Only process dictionaries
                    cleaned_details = {k: (v if not (isinstance(v, float) and np.isnan(v)) else "N/A") for k, v in details.items()}
                    cleaned_day[meal] = cleaned_details
                else:
                    cleaned_day[meal] = details  # Keep "N/A" or other non-dict values as is
            cleaned_plan.append(cleaned_day)
        return cleaned_plan

    clean_diet_plan = clean_data(diet_plan)
    clean_exercise_plan = clean_data(exercise_plan)

    return jsonify({
        "diet_plan": clean_diet_plan,
        "exercise_plan": clean_exercise_plan
    })

if __name__ == '__main__':
    app.run(debug=True)