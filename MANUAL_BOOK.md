# 📖 User Guide (Manual Book)
## Smart Fitness & Food Tracker

Welcome to the **Smart Fitness & Food Tracker**! This application is designed to help you monitor your daily nutrition, log your workouts, analyze your fitness level using Machine Learning, forecast future weight trends, and consult directly with a smart health assistant (AI Chatbot).

This guide will walk you step-by-step through using all the features of the application, starting from when you first open the app.

---

## Table of Contents
1. [Getting Started: Account & Profile Setup](#1-getting-started-account--profile-setup)
2. [Main Menu Navigation](#2-main-menu-navigation)
   - [Dashboard (Home)](#-dashboard-home)
   - [Food Log (Food Tracking & Apriori Analysis)](#-food-log-food-tracking--apriori-analysis)
   - [Activity Log (Workout Tracking)](#-activity-log-workout-tracking)
   - [Fitness Level Classifier](#-fitness-level-classifier)
   - [AI Chatbot (FitBot)](#-ai-chatbot-fitbot)
   - [ML Predictor (Calories Burned Prediction)](#-ml-predictor-calories-burned-prediction)
   - [About](#-about)
3. [Additional Tips for Best Results](#3-additional-tips-for-best-results)

---

## 1. Getting Started: Account & Profile Setup

When you first open the application, you will be greeted by the Authentication Gate. To access the tracker, you must either log in to an existing account or register a new one.

### How to Log In:
1. If you already have an account, stay on the **🔑 Login** tab.
2. Enter your **Username** and **Password**.
3. Click the **🔑 Login** button. Once authenticated, you will be redirected to the Dashboard.

### How to Register & Set Up Your Profile:
If you are a new user, switch to the **📝 Register** tab and complete the multi-step setup wizard:

1. **Step 0 - Welcome**: Click the **🚀 Get Started** button.
2. **Step 1 - Account Details**: Create a unique **Username** and a secure **Password**.
3. **Step 2 - Personal Data**: Provide your **Name**, **Gender** (Male/Female), **Age**, **Height (cm)**, and **Weight (kg)**.
4. **Step 3 - Fitness Parameters**: Select your daily **Activity Level** and your primary **Fitness Goal** (Weight Loss / Maintain Weight / Muscle Gain).
5. **Step 4 - Review Metrics**: Based on your inputs, the system will instantly calculate your health metrics:
   * **BMR (Basal Metabolic Rate)**: Minimum calories your body needs to survive at rest.
   * **TDEE (Total Daily Energy Expenditure)**: Estimated calories burned in a day with physical activity.
   * **Daily Target**: Your specific daily calorie target tailored to your goal.
   * **BMI (Body Mass Index)**: Your body mass index and category.
6. Click **✨ Complete Setup**. Your account will be created, and you will be automatically logged in and taken to the Dashboard.

---

## 2. Main Menu Navigation
Once your profile is successfully saved, you can start using all the application's features by selecting the menus in the left navigation bar (Sidebar):

### 🏠 Dashboard (Home)
This page provides an instant summary of your daily activities:
1. **Daily Metrics**:
   * **Calories In**: The number of calories you have consumed today compared to your daily target.
   * **Calories Out**: The number of calories you have burned through exercise today.
   * **Net Balance**: The difference between calories in and calories out, as well as the remaining calories you can still consume. *Note: You will receive automated excessive/deficient calorie warnings here if your intake is too high or deficit is dangerously low.*
   * **BMI**: Your current Body Mass Index category.
2. **Weekly Calorie Summary**: An interactive bar chart comparing calories in vs. calories out over the last 7 days.
3. **Calorie Trend Analysis**: An interactive line chart displaying the 30-day trend of your Calories In, Calories Out, and Net Calories to help you identify long-term patterns.
4. **Weight Progress**: 
   * Displays a line chart of your weight fluctuations over time.
   * Enter your latest weight in the column on the right and click **📝 Record Weight** to update your weight log.
4. **Weight Forecasting (ML — Linear Regression)**:
   * Predicts your future weight (7, 14, or 30 days ahead) based on your historical weight logs using a *Linear Regression* algorithm.
   * **Note**: This prediction feature requires a minimum of **2 weight logs** on different dates to generate a trend estimate.
5. **Recent Activities**: Displays a list of your most recently logged meals (`Recent Meals`) and physical exercises (`Recent Workouts`) for the day.

---

### 🍎 Food Log (Food Tracking & Apriori Analysis)
On this page, you can log your meals and analyze your eating patterns. There are 4 menu tabs:

#### 1. Tab `📝 Log Food` (Smart Food Logging)
Logging food is very easy because the system supports natural language descriptions.
* **How to Use**:
  1. Describe your meal in the text area. Example: `"2 slices of pizza"`, `"200g chicken breast"`, `"banana"`, or `"rice"`.
  2. Select the meal type under **Meal Type** (Breakfast, Lunch, Dinner, or Snack).
  3. Click **📝 Log Food**. The system will automatically detect the food name and its nutritional content (calories, protein, carbs, fat).

#### 2. Tab `🍽️ Meal Suggestions`
* Displays your remaining daily calorie target.
* Provides healthy food suggestions suitable for consumption based on the remaining calories available.

#### 3. Tab `📋 Food History`
* Displays your food consumption history over the last 30 days in a detailed table format.
* Shows your total calorie intake over the past month.

#### 4. Tab `🛒 Food Association Rules` (Eating Pattern Analysis - Apriori)
This feature analyzes patterns of foods you frequently consume together using the **Apriori Algorithm**.
* **How to Use**:
  1. **Data Source**: Select *"Sample Food History Dataset (Demo)"* for a quick simulation, or *"My Personal Food Logs"* to analyze your own food history.
  2. **Transaction Grouping**: Group transactions by day or by meal type.
  3. **Minimum Support & Confidence**: Adjust the algorithm's sensitivity.
  4. Click **🚀 Analyze Patterns**.
  5. The analysis results will display a list of food association rules (e.g., *"If you eat Bread, you are highly likely to also eat Jam"*), along with a visual of the **Top 10 Most Popular Foods**.

---

### 🏃 Activity Log (Workout Tracking)
Log your physical exercises to track the calories you burn. There are 2 menu tabs:

#### 1. Tab `📝 Log Activity`
* **How to Use**:
  1. Select the activity type under **Activity Type** (e.g., Running, Cycling, Swimming, Yoga, Weight Lifting).
  2. Enter the workout duration in minutes.
  3. Select the intensity level (Low, Medium, High).
  4. The **Estimated Calories Burned** section will automatically calculate the calories burned based on standard MET (Metabolic Equivalent of Task) values and your body weight.
  5. Click the **✅ Log Activity** button to save the activity.

#### 2. Tab `📋 Activity History`
* Displays your entire workout history over the last 30 days.

---

### 🏋️ Fitness Level Classifier
Use this feature to classify your physical fitness level into **A (Excellent), B (Good), C (Average), or D (Poor)** using Machine Learning models.

* **How to Use**:
  1. Enter your physical data: Gender, Age, Height, Weight, and Body Fat Percentage.
  2. Enter your physical fitness test results:
     * Blood pressure (Diastolic & Systolic).
     * Grip strength (`Grip Force`).
     * Body flexibility (`Sit and Bend`).
     * Number of Sit-ups performed in one minute.
     * Longest jump distance (`Broad Jump`).
  3. Choose the Machine Learning model you want to use (Random Forest, XGBoost, or SVM).
  4. Click **🔍 Predict Fitness Level**.
  5. The classification result, the model's confidence level, and supporting exercise recommendations will be displayed on the screen.

---

### 🤖 AI Chatbot (FitBot)
FitBot is an interactive AI-powered virtual assistant ready to answer all your questions regarding fitness and nutrition.

* **How to Use**:
  1. Type your question in the chat input box at the bottom (supports Indonesian and English).
  2. Press Enter. FitBot will analyze your fitness profile and provide personalized advice tailored specifically for you.
  3. You can also click the quick question buttons in the right sidebar (**💡 Quick Questions**) such as *"Give me a home workout plan"* for instant consultation.

---

### 📊 ML Predictor (Calories Burned Prediction)
Uses the **Random Forest Regressor** Machine Learning model to specifically predict calories burned based on the physiological parameters of your workout.

* **How to Use**:
  1. Enter your personal data (Gender, Age, Height, Weight).
  2. Enter your workout session details: workout duration (minutes), average heart rate (bpm), and body temperature (°C).
  3. Click **🔮 Predict Calories Burned** to generate the estimated calories burned.

---

### ℹ️ About
This menu contains background information about the application, the scientific formulas used (Mifflin-St Jeor BMR, TDEE, MET Formula), and the availability status of the Machine Learning model datasets on the application server.

---

## 3. Additional Tips for Best Results
* **Log Consistently**: Log your food and activities every day to keep your progress charts accurate.
* **Weigh Yourself Regularly**: Update your weight once a week on the **Dashboard** to activate and improve the accuracy of the *Weight Forecasting* feature.
* **Use the AI Assistant**: Do not hesitate to ask **FitBot** for healthy recipes or new workout plan variations directly.

---
*We hope you enjoy your healthy lifestyle journey and successfully reach your goals!* 💪
