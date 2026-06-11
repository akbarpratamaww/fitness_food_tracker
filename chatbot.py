import os
import streamlit as st          # <--- BARU: import streamlit
from dotenv import load_dotenv
from groq import Groq
from datetime import datetime

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

            # Add conversation history (max last 8 messages = 4 exchanges)
            # Keeping history short is critical to stay within daily token limits
            if history and len(history) > 0:
                recent_history = history[-8:]
                for msg in recent_history:
                    if msg['role'] in ['user', 'assistant'] and msg.get('content'):
                        messages.append({"role": msg['role'], "content": msg['content']})

            # Add current user message
            messages.append({"role": "user", "content": user_message})

            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "log_food",
                        "description": "ONLY call this if the user has explicitly confirmed (e.g., replied 'Ya', 'Yes', 'Catat', 'Tolong' to your confirmation question) that they want to log a food they ate. DO NOT call this for recommendations, recipes, or without asking first.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "food_name": {"type": "string", "description": "Name of the food"},
                                "calories": {"type": "number", "description": "Estimated total calories (kcal)"},
                                "protein": {"type": "number", "description": "Estimated total protein (g)"},
                                "carbs": {"type": "number", "description": "Estimated total carbohydrates (g)"},
                                "fat": {"type": "number", "description": "Estimated total fat (g)"},
                                "meal_type": {"type": "string", "enum": ["Breakfast", "Lunch", "Dinner", "Snack"], "description": "Type of meal"}
                            },
                            "required": ["food_name", "calories", "protein", "carbs", "fat", "meal_type"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "log_activity",
                        "description": "ONLY call this if the user has explicitly confirmed (e.g., replied 'Ya', 'Yes', 'Catat', 'Tolong' to your confirmation question) that they want to log a workout they did. DO NOT call this for recommendations, workout advice, or without asking first.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "activity_type": {"type": "string", "description": "Name of the exercise or activity"},
                                "duration_minutes": {"type": "number", "description": "Duration in minutes"},
                                "calories_burned": {"type": "number", "description": "Estimated calories burned"},
                                "intensity": {"type": "string", "enum": ["Low", "Medium", "High"], "description": "Intensity of the activity"}
                            },
                            "required": ["activity_type", "duration_minutes", "calories_burned", "intensity"]
                        }
                    }
                }
            ]

            response = self.client.chat.completions.create(
                # llama-3.1-8b-instant: 500k tokens/day free (vs 14k for 70b)
                # Much higher daily quota means far fewer rate-limit errors
                model="llama-3.1-8b-instant",
                messages=messages,
                temperature=0.7,
                max_tokens=350,
                top_p=0.9,
                tools=tools,
                tool_choice="auto"
            )
            
            response_message = response.choices[0].message
            
            # Check for tool calls
            if response_message.tool_calls:
                import json
                from datetime import date
                from database import add_food_log, add_activity_log
                
                tool_call = response_message.tool_calls[0]
                function_name = tool_call.function.name
                
                try:
                    function_args = json.loads(tool_call.function.arguments)
                except:
                    function_args = {}
                    
                user_id = self.user_data.get('user_id') if self.user_data else None
                tool_result_text = "Data not logged."
                
                if user_id:
                    if function_name == "log_food":
                        add_food_log(
                            user_id=user_id,
                            food_name=function_args.get("food_name", "Unknown Food"),
                            calories=function_args.get("calories", 0),
                            protein=function_args.get("protein", 0),
                            carbs=function_args.get("carbs", 0),
                            fat=function_args.get("fat", 0),
                            meal_type=function_args.get("meal_type", "Snack"),
                            log_date=date.today().isoformat()
                        )
                        tool_result_text = f"SUCCESS: Logged {function_args.get('food_name')} with {function_args.get('calories', 0):.0f} kcal."
                    
                    elif function_name == "log_activity":
                        add_activity_log(
                            user_id=user_id,
                            activity_type=function_args.get("activity_type", "Exercise"),
                            duration_minutes=function_args.get("duration_minutes", 0),
                            calories_burned=function_args.get("calories_burned", 0),
                            intensity=function_args.get("intensity", "Medium"),
                            log_date=date.today().isoformat()
                        )
                        tool_result_text = f"SUCCESS: Logged {function_args.get('activity_type')} burning {function_args.get('calories_burned', 0):.0f} kcal."

                # Append the assistant's tool call message
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments
                            }
                        }
                    ]
                })
                
                # Append the tool result message
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": tool_result_text
                })
                
                # Make a second API call to get the final conversational response
                second_response = self.client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=350,
                    top_p=0.9
                )
                
                return self._clean_response(second_response.choices[0].message.content)

            return self._clean_response(response_message.content)

        except Exception as e:
            return self._handle_api_error(e)

    def _handle_api_error(self, e):
        """Convert technical API errors into friendly user-facing messages."""
        error_str = str(e).lower()

        # Rate limit / quota exceeded
        if '429' in str(e) or 'rate_limit' in error_str or 'rate limit' in error_str or 'tokens per day' in error_str or 'tpd' in error_str:
            import re
            # Try to extract wait time from error message
            wait_match = re.search(r'try again in (\d+m\d+\.?\d*s|\d+\.?\d*s|\d+ minute)', str(e), re.IGNORECASE)
            wait_info = f" Coba lagi dalam sekitar **{wait_match.group(1)}**." if wait_match else " Coba lagi dalam beberapa menit."
            return (
                "⏳ **FitBot sedang istirahat sebentar!**\n\n"
                "AI Coach kita sudah banyak membantu hari ini dan perlu jeda sejenak. "
                f"{wait_info}\n\n"
                "💡 Sementara menunggu, kamu bisa:\n"
                "- Lihat progres di menu **Dashboard**\n"
                "- Catat makanan di menu **Food Log**\n"
                "- Catat olahraga di menu **Activity Log**"
            )

        # Authentication / API key invalid
        if '401' in str(e) or 'invalid api key' in error_str or 'unauthorized' in error_str or 'authentication' in error_str:
            return (
                "🔑 **FitBot tidak dapat terhubung saat ini.**\n\n"
                "Terjadi masalah konfigurasi pada layanan AI. "
                "Silakan hubungi admin aplikasi untuk membantu."
            )

        # Network / connection error
        if 'timeout' in error_str or 'connection' in error_str or 'network' in error_str or 'unreachable' in error_str:
            return (
                "🌐 **Koneksi bermasalah.**\n\n"
                "FitBot tidak dapat terhubung ke server AI saat ini. "
                "Pastikan koneksi internet kamu stabil, lalu coba kirim pesan lagi."
            )

        # Model overloaded / server error
        if '503' in str(e) or '500' in str(e) or 'overloaded' in error_str or 'service unavailable' in error_str:
            return (
                "🛠️ **Server AI sedang sibuk.**\n\n"
                "Terlalu banyak pengguna mengakses FitBot sekarang. "
                "Tunggu sebentar dan coba lagi ya! 🙏"
            )

        # Generic fallback — still friendly, no raw details
        return (
            "😅 **FitBot mengalami kendala teknis.**\n\n"
            "Maaf atas ketidaknyamanannya! Silakan coba kirim pesan lagi. "
            "Jika masalah terus berlanjut, coba refresh halaman."
        )

    def _clean_response(self, text):
        """Remove any raw function call syntax that leaked into the text response."""
        if text is None:
            return ""
        import re
        # Strip <function=xxx>{...}</function> patterns
        cleaned = re.sub(r'<function=\w+>.*?</function>', '', text, flags=re.DOTALL)
        # Strip ```json {...} ``` blocks that look like function args
        cleaned = re.sub(r'```json\s*\{[^`]*\}\s*```', '', cleaned, flags=re.DOTALL)
        cleaned = cleaned.strip()
        # If after cleaning the response is empty, return a generic confirmation
        if not cleaned:
            cleaned = "✅ Sudah dicatat ke log Anda!"
        return cleaned

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

        prompt = f"""You are FitBot, a friendly fitness & nutrition coach.

USER: {self.user_data.get('name', 'User')}, {self.user_data.get('age', '?')}yo {self.user_data.get('gender', '')}, {self.user_data.get('weight_kg', '?')}kg/{self.user_data.get('height_cm', '?')}cm, BMI {bmi:.1f} ({bmi_cat})
Goal: {self.user_data.get('fitness_goal', 'Maintain Weight')} | Target: {self.user_data.get('daily_target_calories', 0):.0f} kcal/day | TDEE: {self.user_data.get('tdee', 0):.0f} kcal

RULES:
- Reply in user's language (Indonesian or English). Be concise (3-5 sentences max + bullets).
- Use user's actual numbers when giving advice. Be specific and encouraging.
- Off-topic questions: politely redirect to fitness/health.
- LOG RULE (STRICT): ALWAYS ask user for confirmation before calling log_food or log_activity. Only call the tool AFTER user explicitly says Ya/Yes/Catat/Iya/Boleh. Never call automatically.
- NEVER output raw text like <function=log_food>...</function>. Use the actual tool call only.

Tone: friendly, motivating, use emojis sparingly (💪🥗🏃)."""

        return prompt

    def _get_default_system_prompt(self):
        return """You are FitBot, a friendly fitness and nutrition coach. 
Provide practical, science-based advice on exercise, diet, weight loss, muscle gain, and healthy habits.
Keep responses concise (under 200 words) and actionable. Use emojis to be engaging.
Respond in the same language as the user (Indonesian or English).

CRITICAL RULE: If the user asks about topics completely unrelated to fitness, nutrition, or health, politely decline to answer initially, stating your expertise is strictly limited to health and fitness. HOWEVER, if the user insists or forces you to answer the unrelated topic, briefly answer it but IMMEDIATELY pivot the conversation back to their fitness goals, diet, or app features.

DATABASE ACCESS: You have direct database access via the `log_food` and `log_activity` tools. Before calling any tool, you MUST always ask the user for text confirmation first. Only call the tool if they explicitly agree (e.g. reply 'Ya/Yes/Catat'). Do NOT call the tool on recommendations or without asking."""

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