import pandas as pd
import numpy as np
from datetime import date, timedelta

# Activity MET values (Metabolic Equivalent of Task)
# Calories burned per minute = MET * 3.5 * weight_kg / 200
ACTIVITY_MET = {
    'Walking (slow)': 2.5,
    'Walking (brisk)': 3.5,
    'Running (5 mph)': 8.3,
    'Running (6 mph)': 9.8,
    'Cycling (leisure)': 5.5,
    'Cycling (moderate)': 8.0,
    'Swimming': 7.0,
    'Yoga': 3.0,
    'Strength Training': 5.0,
    'HIIT': 8.5,
    'Dancing': 5.0,
    'Stair Climbing': 8.0,
    'Elliptical': 6.0,
    'Rowing': 7.0,
    'Jump Rope': 12.0,
    'Resting': 1.0,
}

# Meal type categories
MEAL_TYPES = ['Breakfast', 'Lunch', 'Dinner', 'Snack']

# Activity levels for TDEE calculation
ACTIVITY_MULTIPLIERS = {
    'Sedentary (little or no exercise)': 1.2,
    'Lightly active (light exercise 1-3 days/week)': 1.375,
    'Moderately active (moderate exercise 3-5 days/week)': 1.55,
    'Very active (hard exercise 6-7 days/week)': 1.725,
    'Extra active (very hard exercise & physical job)': 1.9,
}

# Fitness goals
FITNESS_GOALS = {
    'Weight Loss': -500,      # 500 calorie deficit per day
    'Maintain Weight': 0,
    'Muscle Gain': 300,       # 300 calorie surplus per day
}

def calculate_bmr(weight_kg, height_cm, age, gender):
    """
    Calculate Basal Metabolic Rate using Mifflin-St Jeor equation.
    BMR is the number of calories your body burns at complete rest.
    """
    if gender == 'Male':
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
    else:
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161
    return bmr

def calculate_tdee(bmr, activity_level):
    """
    Calculate Total Daily Energy Expenditure.
    TDEE = BMR × Activity Multiplier
    """
    multiplier = ACTIVITY_MULTIPLIERS.get(activity_level, 1.2)
    return bmr * multiplier

def calculate_daily_target(tdee, fitness_goal):
    """Calculate daily calorie target based on fitness goal."""
    adjustment = FITNESS_GOALS.get(fitness_goal, 0)
    return tdee + adjustment

def calculate_bmi(weight_kg, height_cm):
    """Calculate Body Mass Index."""
    height_m = height_cm / 100
    return weight_kg / (height_m ** 2)

def get_bmi_category(bmi):
    """Get BMI category."""
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal weight"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

def calculate_calories_burned_met(activity_type, duration_minutes, weight_kg):
    """
    Calculate calories burned using MET formula.
    Calories burned per minute = (MET * 3.5 * weight_kg) / 200
    """
    met = ACTIVITY_MET.get(activity_type, 3.0)
    calories_per_minute = (met * 3.5 * weight_kg) / 200
    return calories_per_minute * duration_minutes

def parse_food_input(user_input):
    """
    Parse natural language food input.
    Example: "ate 2 slices of pizza" or "200g chicken breast"
    Returns dict with parsed information.
    """
    import re
    
    user_input = user_input.lower().strip()
    
    # Common food calorie database (simplified)
    food_calories = {
        'pizza': 285, 'burger': 354, 'sandwich': 250, 'rice': 130, 'bread': 265,
        'chicken': 165, 'beef': 250, 'fish': 206, 'egg': 155, 'milk': 42,
        'yogurt': 59, 'cheese': 402, 'apple': 52, 'banana': 89, 'orange': 47,
        'salad': 20, 'pasta': 131, 'soup': 75, 'potato': 77, 'fries': 312,
        'cake': 424, 'chocolate': 546, 'ice cream': 207, 'coffee': 1, 'tea': 1,
        'water': 0, 'soda': 41, 'juice': 45, 'protein shake': 120,
    }
    
    # Extract quantity
    quantity_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:g|gram|kg|slice|bowl|cup|piece|serving)', user_input)
    quantity = 1
    if quantity_match:
        quantity = float(quantity_match.group(1))
        # Convert kg to g
        if 'kg' in user_input and quantity < 10:
            quantity = quantity * 1000
    
    # Find food item
    detected_food = None
    for food in food_calories:
        if food in user_input:
            detected_food = food
            break
    
    if detected_food:
        calories_per_100g = food_calories[detected_food]
        # Assume standard serving of ~100g if no quantity specified
        if quantity == 1 and 'g' not in user_input and 'gram' not in user_input:
            quantity = 100
        calories = (calories_per_100g * quantity) / 100
        return {
            'food_name': detected_food.title(),
            'calories': round(calories, 1),
            'protein': round(calories_per_100g * 0.1 * quantity / 100, 1),
            'carbs': round(calories_per_100g * 0.15 * quantity / 100, 1),
            'fat': round(calories_per_100g * 0.05 * quantity / 100, 1),
            'detected': True
        }
    
    return {'detected': False, 'food_name': user_input, 'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0}

def load_food_dataset():
    """
    Load food dataset from CSV file.
    Creates a sample dataset if file doesn't exist.
    """
    try:
        df = pd.read_csv('data/food_dataset.csv')
        return df
    except FileNotFoundError:
        # Create sample food dataset
        sample_foods = pd.DataFrame({
            'Food': ['Apple', 'Banana', 'Orange', 'Chicken Breast', 'Salmon', 'Rice', 'Pasta', 
                     'Bread', 'Egg', 'Milk', 'Yogurt', 'Cheese', 'Broccoli', 'Spinach', 'Carrot',
                     'Pizza', 'Burger', 'French Fries', 'Ice Cream', 'Chocolate Cake'],
            'Calories_per_100g': [52, 89, 47, 165, 208, 130, 131, 265, 155, 42, 59, 402, 34, 23, 41,
                                   285, 354, 312, 207, 424],
            'Protein_g': [0.3, 1.1, 0.9, 31, 20, 2.7, 5, 9, 13, 3.4, 10, 25, 2.8, 2.9, 0.9,
                          12, 17, 3.4, 3.5, 5.3],
            'Carbs_g': [14, 23, 12, 0, 0, 28, 25, 49, 1.1, 5, 3.6, 1.3, 7, 3.6, 10,
                        30, 30, 41, 24, 58],
            'Fat_g': [0.2, 0.3, 0.1, 3.6, 13, 0.3, 1.1, 3.2, 11, 1, 0.4, 33, 0.4, 0.4, 0.2,
                      10, 20, 15, 11, 20]
        })
        sample_foods.to_csv('data/food_dataset.csv', index=False)
        return sample_foods

def generate_meal_recommendation(calories_remaining, diet_type=None):
    """Generate meal recommendations based on remaining calories."""
    recommendations = []
    
    if calories_remaining > 500:
        recommendations.append({
            'type': 'High Protein Meal',
            'options': ['Grilled chicken with quinoa and vegetables (~450 cal)',
                       'Salmon with sweet potato (~500 cal)',
                       'Tofu stir-fry with brown rice (~420 cal)']
        })
    elif calories_remaining > 200:
        recommendations.append({
            'type': 'Light Meal',
            'options': ['Greek yogurt with berries (~200 cal)',
                       'Protein smoothie (~250 cal)',
                       'Turkey sandwich on whole wheat (~280 cal)']
        })
    else:
        recommendations.append({
            'type': 'Healthy Snack',
            'options': ['Apple with peanut butter (~150 cal)',
                       'Handful of almonds (~170 cal)',
                       'Cottage cheese (~100 cal)']
        })
    
    return recommendations

def generate_workout_recommendation(fitness_goal):
    """Generate workout recommendations based on fitness goal."""
    recommendations = {
        'Weight Loss': {
            'focus': 'Cardio and HIIT for maximum calorie burn',
            'suggestions': [
                '30 min HIIT session (burns ~300-400 calories)',
                '45 min brisk walking (burns ~200-250 calories)',
                '20 min jump rope (burns ~250 calories)'
            ]
        },
        'Muscle Gain': {
            'focus': 'Strength training with progressive overload',
            'suggestions': [
                'Upper body strength training (45 min)',
                'Lower body strength training (45 min)',
                'Compound lifts: squats, deadlifts, bench press'
            ]
        },
        'Maintain Weight': {
            'focus': 'Balanced routine for health maintenance',
            'suggestions': [
                '30 min moderate cardio + 20 min strength',
                'Active rest day: yoga or stretching',
                'Sports activity: swimming, cycling, or dancing'
            ]
        }
    }
    
    return recommendations.get(fitness_goal, recommendations['Maintain Weight'])