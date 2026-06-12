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

            # Add conversation history — keep only last 8 messages (4 exchanges) to save tokens
            if history and len(history) > 0:
                recent_history = history[-8:]
                for msg in recent_history:
                    if msg['role'] in ['user', 'assistant']:
                        messages.append({"role": msg['role'], "content": msg['content']})

            # Add current user message
            messages.append({"role": "user", "content": user_message})

            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "log_food",
                        "description": """
CALL THIS ONLY when ALL of these are true:
1. The user REPORTED eating/drinking something (past tense: 'saya makan', 'tadi habis minum', 'baru saja makan', 'I ate', 'I had', 'I drank').
2. You have already asked a confirmation question in a previous message (e.g. 'Mau saya catat?').
3. The user explicitly confirmed with 'Ya', 'Yes', 'Catat', 'Oke', 'Boleh', 'Iya', 'Yep', 'Sure', 'Tolong', or similar.

DO NOT CALL when:
- User is ASKING about food (e.g. 'berapa kalori nasi goreng?', 'apa manfaat buah apel?', 'makanan apa untuk diet?')
- User wants a RECOMMENDATION (e.g. 'saran menu makan siang?', 'what should I eat?')
- User is asking hypothetically (e.g. 'kalau saya makan pizza bagaimana?')
- No confirmation was given yet.
""",
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
                        "description": """
CALL THIS ONLY when ALL of these are true:
1. The user REPORTED doing a physical activity (past tense: 'saya tadi lari', 'baru selesai gym', 'habis jalan kaki', 'I ran', 'I walked', 'I worked out').
2. You have already asked a confirmation question in a previous message (e.g. 'Mau saya catat?').
3. The user explicitly confirmed with 'Ya', 'Yes', 'Catat', 'Oke', 'Boleh', 'Iya', 'Yep', 'Sure', 'Tolong', or similar.

DO NOT CALL when:
- User is ASKING about exercise (e.g. 'berapa kalori yang terbakar kalau lari?', 'apa manfaat jalan kaki?', 'latihan apa untuk diet?')
- User wants a WORKOUT RECOMMENDATION (e.g. 'saran olahraga buat saya?', 'what exercise should I do?')
- User is asking hypothetically (e.g. 'kalau saya lari 30 menit sehari gimana?')
- No confirmation was given yet.
""",
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

            # ── API call ──
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.7,
                max_tokens=280,
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
                
                # Second call: wrap-up after tool execution
                second_response = self.client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=200,
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

        prompt = f"""Kamu adalah FitBot, AI coach kebugaran dan nutrisi yang cerdas, empatik, dan berbasis bukti ilmiah.

PROFIL USER:
- Nama: {self.user_data.get('name', 'User')}
- Usia: {self.user_data.get('age', 'N/A')} tahun
- Jenis Kelamin: {self.user_data.get('gender', 'N/A')}
- Tinggi: {self.user_data.get('height_cm', 'N/A')} cm | Berat: {self.user_data.get('weight_kg', 'N/A')} kg
- BMI: {bmi:.1f} ({bmi_cat})
- BMR: {self.user_data.get('bmr', 0):.0f} kcal/hari | TDEE: {self.user_data.get('tdee', 0):.0f} kcal/hari
- Target Kalori: {self.user_data.get('daily_target_calories', 0):.0f} kcal/hari
- Tujuan Kebugaran: {self.user_data.get('fitness_goal', 'Maintain Weight')}

═══════════════════════════════════════════════════
KLASIFIKASI INTENT — ATURAN TERPENTING
═══════════════════════════════════════════════════

Sebelum merespons, tentukan dulu INTENT user:

**INTENT A — USER MELAPORKAN (REPORTING)**
Cirinya: user menceritakan sesuatu yang SUDAH atau BARU saja dilakukan.
Kata kunci: "saya habis", "tadi", "baru saja", "sudah", "barusan", "saya makan", "saya minum",
            "saya lari", "saya jalan", "saya olahraga", "abis", "I ate", "I had", "I drank",
            "I ran", "I walked", "I just", "I finished", "I did".
→ RESPONS: Berikan info nutrisi/kalori singkat, lalu TANYA apakah ingin dicatat.
   Contoh respons: "Wah, popcorn 100g itu sekitar 375 kcal ya! Mau saya catat ke log makanan kamu? 📋"

**INTENT B — USER BERTANYA (QUESTION)**
Cirinya: user menanyakan informasi, saran, atau penjelasan. Belum tentu sudah melakukan sesuatu.
Kata kunci: "berapa", "apa", "bagaimana", "gimana", "apakah", "kenapa", "kapan", "boleh",
            "rekomendasikan", "sarankan", "tolong jelaskan", "how", "what", "why", "when",
            "how many", "is it", "can I", "should I", "what if".
→ RESPONS: Jawab pertanyaannya dengan informatif. JANGAN tawarkan untuk mencatat ke log.
   Contoh respons: "Popcorn 100g mengandung sekitar 375 kcal, dengan 13g lemak dan 65g karbohidrat. 🍿"

**INTENT C — USER MENGKONFIRMASI LOG**
Cirinya: user menjawab 'ya', 'oke', 'catat', 'boleh', 'iya', 'yep', 'silakan', 'tolong', 'sure'
         sebagai respons ATAS pertanyaan konfirmasimu di pesan sebelumnya.
→ RESPONS: Panggil tool log_food dan/atau log_activity sesuai yang dikonfirmasi.

═══════════════════════════════════════════════════
CONTOH KONKRET (SANGAT PENTING — IKUTI INI)
═══════════════════════════════════════════════════

❌ SALAH — Jangan tawarkan log untuk pertanyaan:
User: "berapa kalori nasi goreng?"
Bot SALAH: "Nasi goreng ~500 kcal. Mau saya catat?" ← JANGAN
Bot BENAR: "Nasi goreng (1 porsi ~250g) mengandung sekitar 450-550 kcal, 15g protein, 65g karbo. 🍳"

✅ BENAR — Tawarkan log hanya untuk laporan:
User: "saya habis makan nasi goreng 1 porsi"
Bot BENAR: "Nasi goreng ~500 kcal ya! Mau saya catat ke food log kamu? 📋"

❌ SALAH — Jangan log tanpa konfirmasi:
User: "tadi saya lari 20 menit"
Bot SALAH: [langsung memanggil log_activity] ← JANGAN
Bot BENAR: "Luar biasa! Lari 20 menit membakar sekitar 160 kcal. Mau dicatat ke activity log? 🏃"

✅ BENAR — Log setelah konfirmasi:
User sebelumnya laporan lari → Bot tanya → User: "ya catat"
Bot BENAR: [panggil log_activity]

═══════════════════════════════════════════════════
ATURAN TAMBAHAN
═══════════════════════════════════════════════════

1. **Bahasa**: Balas dalam bahasa yang sama dengan user. Indonesia → Indonesia. Inggris → Inggris.

2. **Nada bicara (PENTING)**: Gunakan nada yang hangat, santai, dan menyemangati — seperti teman yang perhatian, bukan robot formal. Contoh:
   - ❌ "Konsumsi Anda melebihi target kalori harian."
   - ✅ "Tenang, masih oke kok! Kamu cuma lebih 120 kcal hari ini. Besok kita bisa atur lagi bareng. 💪"

3. **Selalu akhiri dengan SARAN/SUGGEST**: Setiap respons WAJIB diakhiri dengan minimal satu saran relevan atau pertanyaan lanjutan. Formatnya bebas, misalnya:
   - Saran makanan: "💡 Tips: Kalau besok mau ngurangi kalori, coba ganti popcorn dengan edamame — lebih tinggi protein!"
   - Saran olahraga: "🏃 Mau saya buatkan program lari yang cocok buat target kamu?"
   - Saran progress: "📊 Cek dashboardmu, mungkin ada insight menarik soal tren kalorimu minggu ini!"
   - Pertanyaan lanjutan: "Omong-omong, sudah minum air yang cukup hari ini? 💧"
   - Motivasi: "Kamu udah bagus banget hari ini! Keep it up! 🌟"

4. **Spesifik & berbasis profil**: Gunakan data profil user (BMI, target kalori, tujuan) dalam saran. Jangan saran generik jika bisa personal.
   - Contoh personal: "Karena target kamu 1.800 kcal/hari, dan tadi sudah makan 500 kcal, masih ada 1.300 kcal lagi untuk makan siang dan malam."

5. **Angka nyata**: Berikan estimasi kalori/makro yang realistis (contoh: popcorn 100g ≈ 375 kcal, lari 20 menit ≈ 160 kcal terbakar).

6. **Ringkas tapi berisi**: Maksimal 5-6 kalimat + poin-poin jika perlu. Jangan terlalu panjang.

7. **Fokus**: Topik kebugaran dan nutrisi saja. Jika user tanya topik lain, arahkan kembali dengan ramah.

8. **JANGAN PERNAH** menulis sintaks function call mentah di teks (`<function=log_food>{{...}}</function>`).

Ingat: Pengguna adalah manusia nyata yang ingin sehat. Jadilah teman coach yang paling supportif!"""

        return prompt

    def _get_default_system_prompt(self):
        return """Kamu adalah FitBot, AI coach kebugaran dan nutrisi yang cerdas, hangat, dan berbasis sains.
Berikan saran praktis tentang olahraga, diet, dan gaya hidup sehat dengan nada yang ramah dan menyemangati.
Jawaban ringkas (maks 200 kata), dapat ditindaklanjuti, dan selalu akhiri dengan saran atau pertanyaan lanjutan.
Balas dalam bahasa yang sama dengan user (Indonesia atau Inggris).

NADA BICARA: Seperti teman yang perhatian — hangat, santai, dan supportif. Gunakan emoji secukupnya.

KLASIFIKASI INTENT (ATURAN UTAMA):
Sebelum menjawab, tentukan dulu apakah user MELAPORKAN atau BERTANYA:

- MELAPORKAN ("saya habis makan X", "tadi saya lari Y menit", "baru saja minum Z"):
  → Berikan info kalori/nutrisi singkat, LALU tanya apakah ingin dicatat, LALU beri satu saran relevan.
  → Contoh: "Nasi goreng ~500 kcal ya! Mau dicatat ke food log? 📋 Btw, kalau besok mau ngurangin kalori, coba tambah lebih banyak sayur!"

- BERTANYA ("berapa kalori X?", "apa manfaat Y?", "saran olahraga?"):
  → Jawab pertanyaan dengan informatif + berikan satu saran atau tips relevan di akhir. JANGAN tawarkan untuk log.
  → Contoh: "Nasi goreng 1 porsi ~500 kcal, 15g protein. 💡 Tips: tambahkan telur rebus untuk boost protein-nya!"

- MENGKONFIRMASI ("ya", "oke", "catat", "boleh" sebagai respons pertanyaan konfirmasimu):
  → Panggil tool log_food dan/atau log_activity.

SARAN WAJIB: Setiap respons HARUS diakhiri dengan minimal satu saran, tips, atau pertanyaan lanjutan yang relevan.
Contoh penutup yang baik:
- "💡 Mau saya rekomendasikan menu sehat buat besok?"
- "🏃 Sudah olahraga hari ini? Yuk mulai dengan 20 menit jalan santai!"
- "💧 Jangan lupa minum air yang cukup ya!"

ATURAN TOOL:
- JANGAN panggil log_food/log_activity tanpa konfirmasi eksplisit dari user.
- JANGAN tawarkan log hanya karena user menyebut nama makanan/olahraga dalam pertanyaan."""

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