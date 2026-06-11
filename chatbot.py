import os
import re
import json
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from datetime import datetime, date

load_dotenv()

# ── Model Configuration ──────────────────────────────────────────────────────
# llama3-groq-70b-8192-tool-use-preview is Groq's dedicated tool-use model,
# far more reliable at producing proper tool_calls instead of leaking
# function syntax into plain text. Falls back to versatile model if needed.
PRIMARY_MODEL   = "llama3-groq-70b-8192-tool-use-preview"
FOLLOWUP_MODEL  = "llama-3.3-70b-versatile"


class FitnessChatbot:
    def __init__(self, user_data=None):
        self.user_data = user_data
        self.client = None

        api_key = os.getenv('GROQ_API_KEY') or (
            st.secrets.get('GROQ_API_KEY')
            if hasattr(st, 'secrets') else None
        )
        if api_key:
            self.client = Groq(api_key=api_key)

    # ── Public entry point ───────────────────────────────────────────────────
    def get_response(self, user_message, context=None, history=None):
        if self.client:
            return self._get_groq_response(user_message, context, history)
        return self._get_rule_based_response(user_message, context)

    # ── Core Groq response ───────────────────────────────────────────────────
    def _get_groq_response(self, user_message, context, history):
        try:
            system_prompt = self._build_system_prompt()

            if context and context.get('calories_in') is not None:
                system_prompt += (
                    f"\n\n📊 Today's stats: {context['calories_in']:.0f} kcal consumed, "
                    f"{context['calories_out']:.0f} kcal burned. "
                    f"Net: {context['calories_in'] - context['calories_out']:.0f} kcal."
                )

            messages = [{"role": "system", "content": system_prompt}]

            if history:
                for msg in history[-20:]:
                    if msg['role'] in ('user', 'assistant') and msg.get('content'):
                        messages.append({"role": msg['role'], "content": msg['content']})

            messages.append({"role": "user", "content": user_message})

            tools = self._build_tools()

            # ── First API call ───────────────────────────────────────────────
            response = self.client.chat.completions.create(
                model=PRIMARY_MODEL,
                messages=messages,
                temperature=0.6,
                max_tokens=512,
                top_p=0.9,
                tools=tools,
                tool_choice="auto",
            )

            response_message = response.choices[0].message

            # ── Handle proper tool_calls ─────────────────────────────────────
            if response_message.tool_calls:
                tool_call   = response_message.tool_calls[0]
                func_name   = tool_call.function.name
                tool_result = self._execute_tool(func_name, tool_call.function.arguments)

                # Append assistant tool call + tool result for follow-up call
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": func_name,
                            "arguments": tool_call.function.arguments,
                        }
                    }]
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": func_name,
                    "content": tool_result,
                })

                # ── Second API call for conversational reply ─────────────────
                second = self.client.chat.completions.create(
                    model=FOLLOWUP_MODEL,
                    messages=messages,
                    temperature=0.6,
                    max_tokens=512,
                    top_p=0.9,
                )
                final_content = second.choices[0].message.content or ""
            else:
                final_content = response_message.content or ""

            # ── Safety net: strip any leaked <function=...> tags ─────────────
            final_content = self._clean_leaked_functions(final_content)

            return final_content if final_content else \
                "✅ Data berhasil dicatat! Ada lagi yang bisa saya bantu hari ini?"

        except Exception:
            # Log to console for debugging but show friendly message to user
            import traceback
            traceback.print_exc()
            return (
                "Maaf, saya sedang mengalami sedikit gangguan. "
                "Coba tanyakan lagi dalam beberapa detik ya! 😊"
            )

    # ── Tool definitions ─────────────────────────────────────────────────────
    def _build_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "log_food",
                    "description": (
                        "Call this ONLY after the user has explicitly confirmed they want to "
                        "log food (e.g., replied 'Ya', 'Yes', 'Catat', 'Iya', 'Boleh'). "
                        "Never call automatically on recommendations."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "food_name":  {"type": "string",  "description": "Name of the food"},
                            "calories":   {"type": "number",  "description": "Total calories (kcal)"},
                            "protein":    {"type": "number",  "description": "Total protein (g)"},
                            "carbs":      {"type": "number",  "description": "Total carbohydrates (g)"},
                            "fat":        {"type": "number",  "description": "Total fat (g)"},
                            "meal_type":  {
                                "type": "string",
                                "enum": ["Breakfast", "Lunch", "Dinner", "Snack"],
                                "description": "Type of meal"
                            },
                        },
                        "required": ["food_name", "calories", "protein", "carbs", "fat", "meal_type"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "log_activity",
                    "description": (
                        "Call this ONLY after the user has explicitly confirmed they want to "
                        "log an activity (e.g., replied 'Ya', 'Yes', 'Catat'). "
                        "Never call automatically on workout recommendations."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "activity_type":    {"type": "string", "description": "Name of the exercise"},
                            "duration_minutes": {"type": "number", "description": "Duration in minutes"},
                            "calories_burned":  {"type": "number", "description": "Estimated calories burned"},
                            "intensity":        {
                                "type": "string",
                                "enum": ["Low", "Medium", "High"],
                                "description": "Exercise intensity"
                            },
                        },
                        "required": ["activity_type", "duration_minutes", "calories_burned", "intensity"],
                    },
                },
            },
        ]

    # ── Tool execution ───────────────────────────────────────────────────────
    def _execute_tool(self, func_name, arguments_str):
        """Parse arguments and write to DB. Returns a result string."""
        from database import add_food_log, add_activity_log

        try:
            args = json.loads(arguments_str)
        except (json.JSONDecodeError, TypeError):
            return "Error: could not parse tool arguments."

        user_id = self.user_data.get('user_id') if self.user_data else None
        if not user_id:
            return "Error: no user_id found."

        today = date.today().isoformat()

        if func_name == "log_food":
            add_food_log(
                user_id=user_id,
                food_name=args.get("food_name", "Unknown Food"),
                calories=float(args.get("calories", 0)),
                protein=float(args.get("protein", 0)),
                carbs=float(args.get("carbs", 0)),
                fat=float(args.get("fat", 0)),
                meal_type=args.get("meal_type", "Snack"),
                log_date=today,
            )
            return f"SUCCESS: Logged {args.get('food_name')} — {args.get('calories', 0):.0f} kcal."

        elif func_name == "log_activity":
            add_activity_log(
                user_id=user_id,
                activity_type=args.get("activity_type", "Exercise"),
                duration_minutes=float(args.get("duration_minutes", 0)),
                calories_burned=float(args.get("calories_burned", 0)),
                intensity=args.get("intensity", "Medium"),
                log_date=today,
            )
            return (
                f"SUCCESS: Logged {args.get('activity_type')} "
                f"— {args.get('calories_burned', 0):.0f} kcal burned."
            )

        return "Unknown function."

    # ── Leaked function tag cleaner ──────────────────────────────────────────
    def _clean_leaked_functions(self, text):
        """
        Detect and handle any <function=name>{...}</function> that the model
        accidentally leaks into plain text. Executes the tool silently and
        removes the tag from the visible response.
        """
        if "<function=" not in text:
            return text

        # Collect all matches first (findall, not finditer, so we don't exhaust the generator)
        pattern = re.compile(r'<function=([^>]+)>(.*?)</function>', re.DOTALL)
        matches = pattern.findall(text)  # list of (func_name, args_str) tuples

        for func_name, args_str in matches:
            try:
                self._execute_tool(func_name.strip(), args_str.strip())
            except Exception:
                pass  # silently ignore DB errors for leaked calls

        # Strip all leaked tags from the visible text
        cleaned = pattern.sub('', text).strip()
        return cleaned

    # ── System prompt ────────────────────────────────────────────────────────
    def _build_system_prompt(self):
        if not self.user_data:
            return self._get_default_system_prompt()

        h = self.user_data.get('height_cm', 170) / 100
        w = self.user_data.get('weight_kg', 70)
        bmi = w / (h ** 2) if h > 0 else 22
        bmi_cat = (
            "Underweight" if bmi < 18.5 else
            "Normal"      if bmi < 25   else
            "Overweight"  if bmi < 30   else
            "Obese"
        )

        return f"""You are an expert, empathetic fitness & nutrition coach named "FitBot".

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

RESPONSE RULES:
1. Be specific and actionable, referencing the user's actual numbers.
2. Keep answers concise: 3-5 sentences + bullet points when helpful.
3. Respond in the user's language (Indonesian or English).
4. Use emojis occasionally (💪 🥗 🏃 etc.) to keep it friendly.
5. If asked off-topic questions, politely redirect to fitness/nutrition.
6. LOG DATA RULE (STRICT): Before calling log_food or log_activity, ALWAYS ask
   the user for confirmation first (e.g., "Mau saya catat ke log kamu?").
   Only invoke the tool AFTER the user explicitly confirms (Ya/Yes/Catat/Iya/Boleh).
   NEVER call the tool automatically on a first mention or when giving advice.

TONE: Friendly, professional, and motivating."""

    def _get_default_system_prompt(self):
        return """You are FitBot, a friendly fitness and nutrition coach.
Provide concise, actionable advice on exercise, diet, weight loss, and healthy habits.
Respond in the same language as the user (Indonesian or English).
Use emojis to be engaging.

LOG DATA RULE: Before calling log_food or log_activity, ALWAYS ask for user confirmation first.
Only invoke the tool AFTER the user explicitly agrees (Ya/Yes/Catat/Iya/Boleh)."""

    # ── Rule-based fallback (no API) ─────────────────────────────────────────
    def _get_rule_based_response(self, user_message, context):
        msg = user_message.lower().strip()

        if any(w in msg for w in ['lose weight', 'weight loss', 'fat loss', 'turun berat', 'diet turun']):
            if self.user_data:
                deficit = self.user_data.get('tdee', 2500) - self.user_data.get('daily_target_calories', 2000)
                return f"""📉 **Strategi Penurunan Berat Badan**

TDEE kamu {self.user_data.get('tdee', 0):.0f} kcal, target harian {self.user_data.get('daily_target_calories', 0):.0f} kcal — defisit {deficit:.0f} kcal/hari.

✅ **Langkah utama:**
- Konsumsi {self.user_data.get('daily_target_calories', 0):.0f} kcal/hari
- Protein {self.user_data.get('weight_kg', 70)*1.8:.0f}g/hari (1.8g/kg BB)
- Latihan beban 3x/minggu + kardio 150 menit/minggu
- Tidur 7–8 jam

Mau contoh menu makan atau rencana latihan?"""
            return "📉 Buat defisit 300–500 kcal/hari, perbanyak protein & serat, jalan kaki 8.000+ langkah, dan latihan beban 2–3x/minggu. Lengkapi profil agar saya bisa hitung target spesifik kamu!"

        elif any(w in msg for w in ['gain muscle', 'build muscle', 'muscle gain', 'tambah otot', 'bulk']):
            if self.user_data:
                surplus = self.user_data.get('daily_target_calories', 2500) - self.user_data.get('tdee', 2300)
                return f"""💪 **Rencana Muscle Gain**

Maintenance kamu {self.user_data.get('tdee', 0):.0f} kcal. Surplus target: {surplus:.0f} kcal/hari.

✅ **Essensial:**
- Konsumsi {self.user_data.get('daily_target_calories', 0):.0f} kcal (surplus)
- Protein: {self.user_data.get('weight_kg', 70)*1.8:.0f}g/hari
- Progressive overload (tambah beban/rep setiap minggu)
- Compound lift: squat, deadlift, bench press, row
- Tidur 7–9 jam

Mau rencana latihan 3 hari full-body?"""
            return "💪 Untuk membangun otot: makan 200–300 kalori di atas maintenance, latihan beban 3–4x/minggu, protein 1.6–2.2g/kg BB. Lengkapi profil agar saya bisa hitung angka spesifik!"

        elif any(w in msg for w in ['calorie', 'calories', 'kalori', 'how many calories']):
            if context and context.get('calories_in') is not None and self.user_data:
                remaining = self.user_data.get('daily_target_calories', 2000) - context['calories_in']
                return f"""🍽️ **Status Kalori Hari Ini**
- Dikonsumsi: {context['calories_in']:.0f} kcal
- Dibakar: {context['calories_out']:.0f} kcal
- Net: {context['calories_in'] - context['calories_out']:.0f} kcal
- Target: {self.user_data.get('daily_target_calories', 'N/A')} kcal
- Sisa: {remaining:.0f} kcal

💡 Masih lapar? Pilih makanan tinggi volume rendah kalori seperti sayuran, sup, atau Greek yogurt."""
            return "Untuk manajemen kalori: catat semua yang dimakan, prioritaskan makanan utuh, dan log makanan di aplikasi ini agar saya bisa kasih saran spesifik!"

        elif any(w in msg for w in ['workout', 'exercise', 'training', 'olahraga', 'gym', 'latihan']):
            goal = self.user_data.get('fitness_goal', '') if self.user_data else ''
            if goal == 'Weight Loss':
                return "🏋️ **Latihan untuk Turun Berat**\n- 3x/minggu full-body strength (30–40 mnt)\n- 150 mnt kardio sedang (jalan cepat, sepeda)\n- 1x HIIT per minggu\n- Target langkah harian: 8.000+\n\nMau contoh rutinitas spesifik?"
            elif goal == 'Muscle Gain':
                return "🏋️ **Latihan untuk Muscle Gain**\n- Push/Pull/Legs atau Upper/Lower split 4–5x/minggu\n- Fokus compound lift (squat, bench, deadlift, row)\n- 6–12 rep, 3–4 set, progressive overload\n- Istirahat 60–90 detik antar set\n\nMau rencana mingguan spesifik?"
            return "🏃 **Rutinitas Seimbang**\n- 2–3x latihan beban\n- 2–3x kardio (lari, sepeda, renang)\n- 1–2x active recovery (yoga, jalan)\n- Stretching setiap hari\n\nCeritakan peralatan yang kamu punya dan saya design plannya!"

        elif any(w in msg for w in ['meal', 'food', 'diet', 'eat', 'makan', 'menu', 'makanan']):
            if self.user_data and self.user_data.get('daily_target_calories'):
                target = self.user_data.get('daily_target_calories')
                return f"""🥗 **Framework Makan Sehat** (~{target:.0f} kcal/hari)

Sarapan (25%): Oats + buah + protein (~{target*0.25:.0f} kcal)
Makan siang (30%): Ayam/ikan + nasi merah + sayur (~{target*0.3:.0f} kcal)
Snack (15%): Greek yogurt atau buah + kacang (~{target*0.15:.0f} kcal)
Makan malam (30%): Protein + ubi/kentang + brokoli (~{target*0.3:.0f} kcal)

💧 Minum 2–3L air. Sesuaikan porsi dengan rasa lapar.

Mau opsi vegetarian atau menu spesifik masakan Indonesia?"""
            return "Piring sehat: 1/2 sayuran, 1/4 protein lean, 1/4 karbohidrat kompleks. Tambahkan lemak sehat. Log makananmu di app ini untuk saran yang lebih personal!"

        elif any(w in msg for w in ['motivasi', 'motivation', 'motivate', 'give up', 'nyerah', 'stuck']):
            return """🌟 **Kamu pasti bisa!**

Ingat: setiap langkah kecil itu berarti.
- Progres, bukan kesempurnaan
- Miss workout hari ini? Mulai lagi besok
- Kamu sedang membangun kebiasaan seumur hidup 💪

Satu hal apa yang bisa kamu lakukan hari ini untuk maju?"""

        elif any(w in msg for w in ['sleep', 'rest', 'recovery', 'tidur', 'istirahat']):
            return """😴 **Tips Tidur & Pemulihan**
- Targetkan 7–9 jam tidur berkualitas
- Jam tidur dan bangun konsisten
- Hindari layar 30 menit sebelum tidur
- Active recovery: jalan santai, stretching, foam roll
- Hari istirahat = hari pertumbuhan otot!

Mau rutinitas sebelum tidur yang efektif?"""

        elif any(w in msg for w in ['hello', 'hi', 'hey', 'halo', 'hai', 'pagi', 'siang', 'malam']):
            hour = datetime.now().hour
            greeting = "Selamat pagi! ☀️" if hour < 12 else ("Selamat siang! 🌤️" if hour < 18 else "Selamat malam! 🌙")
            return f"""{greeting} Saya FitBot, pelatih kebugaran dan nutrisi pribadimu!

Saya bisa bantu:
• 🥗 Nutrisi & perencanaan makan
• 💪 Program latihan
• 🔥 Hitung kalori & makro
• 📈 Motivasi & progres
• 😴 Tidur & pemulihan

Apa yang ingin kamu fokuskan hari ini?"""

        return """Hei! Saya di sini untuk mendukung perjalanan kebugaranmu 😊

Coba tanyakan:
- "Bagaimana cara turun berat badan?"
- "Makanan terbaik untuk muscle gain?"
- "Berikan latihan di rumah"
- "Berapa kalori yang harus saya makan?"
- "Saya butuh motivasi!"

Ada pertanyaan spesifik?"""

    def clear_history(self):
        self.conversation_history = []