import os
import streamlit as st          # <--- BARU: import streamlit
from dotenv import load_dotenv
from groq import Groq

load_dotenv()  # tetap digunakan untuk localhost

class FitnessChatbot:
    def __init__(self, user_data=None):
        self.user_data = user_data
        self.client = None
        
        # --- Mencari API Key dari berbagai sumber ---
        api_key = None
        
        # 1. Coba dari environment variable (file .env atau sistem)
        if os.getenv('GROQ_API_KEY'):
            api_key = os.getenv('GROQ_API_KEY')
        
        # 2. Jika tidak ada, coba dari Streamlit Secrets (untuk cloud)
        elif hasattr(st, 'secrets') and 'GROQ_API_KEY' in st.secrets:
            api_key = st.secrets['GROQ_API_KEY']
        
        # 3. Jika ditemukan, inisialisasi client Groq
        if api_key:
            self.client = Groq(api_key=api_key)

    def get_response(self, user_message, context=None, history=None):
        """Generate response using Groq API or fallback rule-based."""
        if self.client:
            return self._get_groq_response(user_message, context, history)
        else:
            return self._get_rule_based_response(user_message, context)

    def _get_groq_response(self, user_message, context, history):
        """Get response from Groq API with full conversation memory."""
        try:
            # Build dynamic system prompt with user profile and context
            system_prompt = self._build_system_prompt()
            
            # Add today's calorie context if available
            if context and context.get('calories_in') is not None:
                system_prompt += f"\n\n📊 Today's stats: {context['calories_in']:.0f} kcal consumed, {context['calories_out']:.0f} kcal burned. Net: {context['calories_in'] - context['calories_out']:.0f} kcal."

            # Prepare messages for API
            messages = [{"role": "system", "content": system_prompt}]

            # Add conversation history (max last 20 messages to avoid token overflow)
            if history and len(history) > 0:
                recent_history = history[-20:]  # keep last 10 exchanges
                for msg in recent_history:
                    if msg['role'] in ['user', 'assistant']:
                        messages.append({"role": msg['role'], "content": msg['content']})

            # Add current user message
            messages.append({"role": "user", "content": user_message})

            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",  # or "gemma2-9b-it" "llama-3.1-8b-instant" "llama-3.3-70b-versatile"
                messages=messages,
                temperature=0.7,
                max_tokens=400,
                top_p=0.9,
                frequency_penalty=0.3,
                presence_penalty=0.3
            )
            return response.choices[0].message.content

        except Exception as e:
            return f"⚠️ Maaf, terjadi kesalahan teknis. Silakan coba lagi. Detail: {str(e)}"

    def _build_system_prompt(self):
        """Build a rich system prompt with user profile and coaching instructions."""
        if not self.user_data:
            return self._get_default_system_prompt()

        # Calculate BMI and get category
        height_m = self.user_data.get('height_cm', 170) / 100
        weight = self.user_data.get('weight_kg', 70)
        bmi = weight / (height_m ** 2) if height_m > 0 else 22
        if bmi < 18.5:
            bmi_cat = "Underweight"
        elif bmi < 25:
            bmi_cat = "Normal"
        elif bmi < 30:
            bmi_cat = "Overweight"
        else:
            bmi_cat = "Obese"

        prompt = f"""You are an expert, empathetic, and highly knowledgeable fitness & nutrition coach named "FitBot". 
Your goal is to help users achieve their health goals with practical, evidence-based advice.

USER PROFILE:
- Name: {self.user_data.get('name', 'User')}
- Age: {self.user_data.get('age', 'N/A')}
- Gender: {self.user_data.get('gender', 'N/A')}
- Height: {self.user_data.get('height_cm', 'N/A')} cm
- Weight: {self.user_data.get('weight_kg', 'N/A')} kg
- BMI: {bmi:.1f} ({bmi_cat})
- BMR: {self.user_data.get('bmr', 0):.0f} kcal/day
- TDEE: {self.user_data.get('tdee', 0):.0f} kcal/day
- Daily Calorie Target: {self.user_data.get('daily_target_calories', 0):.0f} kcal
- Fitness Goal: {self.user_data.get('fitness_goal', 'Maintain Weight')}

INSTRUCTIONS FOR YOUR RESPONSES:
1. **Be extremely helpful and specific** – Give actionable advice (e.g., "Eat 150g chicken breast with quinoa" not just "eat protein").
2. **Use the user's profile** – Reference their BMI, target calories, and goal.
3. **Be encouraging but honest** – If they ask something unrealistic, explain gently.
4. **Keep responses concise but rich** – Aim for 3-5 sentences + bullet points if needed.
5. **Respond in the user's language** – If they write in Indonesian, reply in Indonesian. If English, reply in English.
6. **Include small motivational tips** when relevant.
7. **If asked about calorie/macro calculations**, provide exact numbers based on their profile.

TONE: Friendly, professional, and motivating. Use emojis occasionally to make it lively (💪, 🥗, 🏃, etc.).

Remember: The user is a real person trying to improve their health. Make every response count!"""

        return prompt

    def _get_default_system_prompt(self):
        return """You are FitBot, a friendly fitness and nutrition coach. 
Provide practical, science-based advice on exercise, diet, weight loss, muscle gain, and healthy habits.
Keep responses concise (under 200 words) and actionable. Use emojis to be engaging.
Respond in the same language as the user (Indonesian or English)."""

    def _get_rule_based_response(self, user_message, context):
        """Enhanced fallback responses when API is not available."""
        msg = user_message.lower().strip()
        
        # --- Weight loss ---
        if any(w in msg for w in ['lose weight', 'weight loss', 'fat loss', 'turunan berat badan', 'diet turun']):
            if self.user_data:
                deficit = self.user_data.get('tdee', 2500) - self.user_data.get('daily_target_calories', 2000)
                return f"""📉 **Weight Loss Strategy** (based on your profile)

Your TDEE is {self.user_data.get('tdee', 0):.0f} kcal, and your target is {self.user_data.get('daily_target_calories', 0):.0f} kcal — a deficit of {deficit:.0f} kcal/day.

✅ **Key actions:**
- Eat {self.user_data.get('daily_target_calories', 0):.0f} kcal daily
- Prioritize protein (1.6-2.2g per kg body weight = {self.user_data.get('weight_kg', 70)*1.8:.0f}g/day)
- Strength train 3x/week + 150 min cardio weekly
- Sleep 7-8 hours

Need a sample meal plan or workout routine?"""
            else:
                return """📉 **Weight Loss Basics**
- Create a 300-500 calorie deficit daily
- Eat more protein and fiber
- Walk 8,000-10,000 steps/day
- Strength train 2-3x/week
Would you like me to calculate your specific calorie target? Please complete your profile first."""

        # --- Muscle gain ---
        elif any(w in msg for w in ['gain muscle', 'build muscle', 'muscle gain', 'tambah otot', 'bulk']):
            if self.user_data:
                surplus = self.user_data.get('daily_target_calories', 2500) - self.user_data.get('tdee', 2300)
                return f"""💪 **Muscle Gain Plan** (personalized)

Your maintenance is {self.user_data.get('tdee', 0):.0f} kcal. Target surplus: {surplus:.0f} kcal/day.

✅ **Essentials:**
- Eat {self.user_data.get('daily_target_calories', 0):.0f} kcal (surplus)
- Protein: {self.user_data.get('weight_kg', 70)*1.8:.0f}g/day
- Progressive overload (add weight/reps weekly)
- Compound lifts: squats, deadlifts, bench press, rows
- Sleep 7-9h for recovery

Want a sample 3-day full-body routine?"""
            else:
                return "💪 To gain muscle, eat 200-300 calories above maintenance, train heavy 3-4x/week, and eat 1.6-2.2g protein/kg. Want me to calculate your numbers? Complete your profile first."

        # --- Calorie tracking help ---
        elif any(w in msg for w in ['calorie', 'calories', 'kalori', 'how many calories']):
            if context and context.get('calories_in') is not None:
                remaining = (self.user_data.get('daily_target_calories', 2000) - context['calories_in']) if self.user_data else 0
                return f"""🍽️ **Today's Calorie Status**
- Consumed: {context['calories_in']:.0f} kcal
- Burned: {context['calories_out']:.0f} kcal
- Net: {context['calories_in'] - context['calories_out']:.0f} kcal
- Target: {self.user_data.get('daily_target_calories', 'N/A') if self.user_data else 'N/A'} kcal
- Remaining: {remaining:.0f} kcal

💡 Tip: If you're hungry, choose high-volume low-calorie foods like vegetables, broth-based soups, or Greek yogurt."""
            else:
                return "To manage calories: track everything, use a food scale, and prioritize whole foods. Log your meals in the app and I can give specific advice!"

        # --- Workout recommendations ---
        elif any(w in msg for w in ['workout', 'exercise', 'training', 'olahraga', 'gym']):
            if self.user_data and self.user_data.get('fitness_goal') == 'Weight Loss':
                return "🏋️ **Workout for Weight Loss**\n- 3x/week full-body strength training (30-40 min)\n- 150 min moderate cardio (brisk walking, cycling)\n- HIIT once a week for metabolic boost\n- Daily step goal: 8,000+\n\nNeed a sample routine?"
            elif self.user_data and self.user_data.get('fitness_goal') == 'Muscle Gain':
                return "🏋️ **Workout for Muscle Gain**\n- Push/Pull/Legs or Upper/Lower split 4-5x/week\n- Focus on compound lifts (squat, bench, deadlift, row)\n- 6-12 reps, 3-4 sets, progressive overload\n- Rest 60-90 seconds between sets\n\nWant a specific weekly plan?"
            else:
                return "🏃 **Balanced Workout Routine**\n- 2-3x strength training\n- 2-3x cardio (running, cycling, swimming)\n- 1-2x active recovery (yoga, walking)\n- Stretch daily\n\nTell me your equipment and I'll design a plan!"

        # --- Meal / food advice ---
        elif any(w in msg for w in ['meal', 'food', 'diet', 'eat', 'makan', 'menu']):
            if self.user_data and self.user_data.get('daily_target_calories'):
                target = self.user_data.get('daily_target_calories')
                return f"""🥗 **Healthy Meal Framework** (~{target} kcal/day)

Breakfast (25%): Oats with berries & protein powder (~{target*0.25:.0f} kcal)
Lunch (30%): Grilled chicken, quinoa, roasted veggies (~{target*0.3:.0f} kcal)
Snack (15%): Greek yogurt or apple with peanut butter (~{target*0.15:.0f} kcal)
Dinner (30%): Salmon, sweet potato, steamed broccoli (~{target*0.3:.0f} kcal)

💧 Drink 2-3L water. Adjust portions based on hunger.

Want vegetarian or specific cuisine options?"""
            else:
                return "A healthy plate: 1/2 veggies, 1/4 lean protein, 1/4 complex carbs. Add healthy fats. Log your meals to get personalized feedback!"

        # --- Motivation ---
        elif any(w in msg for w in ['motivation', 'motivate', 'give up', 'stuck', 'demotivated']):
            return """🌟 **You've got this!** 
Every small step counts. Remember:
- Progress, not perfection
- Missed a workout? Just get back tomorrow
- You're building habits that last a lifetime
- Think of how far you've come

What's ONE thing you can do today to move forward? 💪"""

        # --- Sleep/recovery ---
        elif any(w in msg for w in ['sleep', 'rest', 'recovery', 'tidur']):
            return """😴 **Sleep & Recovery Tips**
- Aim for 7-9 hours quality sleep
- Keep consistent bedtime/wake time
- No screens 30 min before bed
- Active recovery: light walk, stretch, foam roll
- Rest days = muscle growth days

Need a bedtime routine suggestion?"""

        # --- Greeting ---
        elif any(w in msg for w in ['hello', 'hi', 'hey', 'halo', 'hai', 'good morning', 'good afternoon']):
            hour = datetime.now().hour
            if hour < 12:
                greeting = "Good morning! ☀️"
            elif hour < 18:
                greeting = "Good afternoon! 🌤️"
            else:
                greeting = "Good evening! 🌙"
            
            return f"""{greeting} I'm FitBot, your personal fitness coach!

I can help with:
• 🥗 Nutrition & meal planning
• 💪 Workout routines (any equipment)
• 🔥 Calorie tracking & math
• 📈 Progress motivation
• 😴 Sleep & recovery

What would you like to focus on today? Just ask!"""

        # --- Default fallback ---
        else:
            return f"""Thanks for reaching out! I'm here to support your fitness journey.

You can ask me about:
- "How do I lose weight?"
- "Best foods for muscle gain"
- "Give me a home workout"
- "How many calories should I eat?"
- "I feel demotivated, help!"

What specific question do you have? 😊"""

    # Optional: method to clear conversation history
    def clear_history(self):
        self.conversation_history = []