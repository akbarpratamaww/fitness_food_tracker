import os
import re
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from datetime import datetime

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
#  LOCAL NLP TOKENIZER
#  Detects food items, quantities, activities, and durations from raw text.
#  This pre-processing layer is model-agnostic — it enriches the system
#  prompt with structured facts so the LLM can focus on reasoning, not parsing.
# ─────────────────────────────────────────────────────────────────────────────

# ── Food keywords (Indonesian + English) ──────────────────────────────────────
FOOD_TOKENS = [
    # Junk / snack
    "pop corn", "popcorn", "keripik", "chips", "biskuit", "biscuit",
    "cokelat", "chocolate", "candy", "permen", "es krim", "ice cream",
    "donat", "donut", "kue", "cake", "roti", "bread", "pizza", "burger",
    "hotdog", "hot dog", "kentang goreng", "french fries", "nugget",
    # Protein
    "ayam", "chicken", "dada ayam", "sate", "daging", "beef", "steak",
    "ikan", "fish", "salmon", "tuna", "udang", "shrimp", "telur", "egg",
    "tahu", "tempe", "tofu", "susu", "milk", "yogurt", "keju", "cheese",
    # Carb
    "nasi", "rice", "mie", "mi", "noodle", "pasta", "spaghetti",
    "kentang", "potato", "ubi", "singkong", "oat", "granola", "sereal", "cereal",
    "roti gandum", "whole wheat",
    # Fruit / veg
    "apel", "apple", "pisang", "banana", "jeruk", "orange", "mangga", "mango",
    "semangka", "watermelon", "anggur", "grape", "sayur", "vegetable",
    "wortel", "carrot", "bayam", "spinach", "brokoli", "broccoli",
    # Drinks
    "kopi", "coffee", "teh", "tea", "jus", "juice", "air", "water",
    "soda", "minuman", "drink", "susu", "milk", "protein shake", "smoothie",
    # Cuisine
    "rendang", "soto", "bakso", "gado-gado", "nasi goreng", "fried rice",
    "mie goreng", "indomie", "siomai",
]

# ── Activity keywords (Indonesian + English) ──────────────────────────────────
ACTIVITY_TOKENS = {
    # key: canonical name, value: list of aliases
    "jalan kaki":   ["jalan", "jalan kaki", "walking", "walk", "berjalan"],
    "lari":         ["lari", "jogging", "joging", "berlari", "running", "run"],
    "bersepeda":    ["sepeda", "bersepeda", "cycling", "cycle", "bike"],
    "renang":       ["renang", "berenang", "swimming", "swim"],
    "gym":          ["gym", "angkat beban", "weight training", "weightlifting", "fitness"],
    "push up":      ["push up", "push-up", "pushup"],
    "sit up":       ["sit up", "sit-up", "situp"],
    "squat":        ["squat", "squat jump"],
    "yoga":         ["yoga"],
    "pilates":      ["pilates"],
    "hiit":         ["hiit", "high intensity", "interval"],
    "zumba":        ["zumba", "aerobik", "aerobics"],
    "futsal":       ["futsal", "sepak bola", "football", "soccer"],
    "basket":       ["basket", "basketball"],
    "badminton":    ["badminton", "bulu tangkis"],
    "tenis":        ["tenis", "tennis"],
    "bela diri":    ["karate", "taekwondo", "boxing", "tinju", "silat", "bela diri"],
    "stretching":   ["stretching", "stretch", "peregangan"],
    "naik tangga":  ["naik tangga", "stairs", "tangga"],
}

# ── Quantity units ─────────────────────────────────────────────────────────────
UNIT_MAP = {
    "gram": "g", "gr": "g", "g": "g",
    "kilogram": "kg", "kg": "kg",
    "ml": "ml", "mililiter": "ml",
    "liter": "L", "l": "L",
    "porsi": "porsi", "serving": "porsi",
    "sendok": "sdm", "sdm": "sdm",
    "piring": "piring", "plate": "piring",
    "buah": "buah", "biji": "biji", "butir": "butir",
    "gelas": "gelas", "cup": "gelas",
    "mangkuk": "mangkuk", "bowl": "mangkuk",
    "bungkus": "bungkus", "pack": "bungkus", "sachet": "bungkus",
    "potong": "potong", "slice": "potong", "lembar": "potong",
}

# ── Duration patterns ──────────────────────────────────────────────────────────
DURATION_PATTERN = re.compile(
    r'(\d+(?:[.,]\d+)?)\s*'
    r'(jam|hour|hours|hr|menit|minute|minutes|min|detik|second|seconds|sec)',
    re.IGNORECASE
)

NUMBER_PATTERN = re.compile(r'\b(\d+(?:[.,]\d+)?)\b')
UNIT_PATTERN   = re.compile(
    r'\b(\d+(?:[.,]\d+)?)\s*(' + '|'.join(re.escape(u) for u in UNIT_MAP) + r')\b',
    re.IGNORECASE
)


def _to_float(s: str) -> float:
    return float(s.replace(',', '.'))


def parse_user_input(text: str) -> dict:
    """
    Tokenise free-text user input and return a structured dict:
    {
        "foods":      [{"name": str, "qty": float|None, "unit": str|None}],
        "activities": [{"name": str, "duration_minutes": float|None}],
        "numbers":    [float],          # any standalone numbers found
        "raw":        str               # original text
    }
    """
    lower = text.lower()
    result = {"foods": [], "activities": [], "numbers": [], "raw": text}

    # ── Detect quantities with units ──────────────────────────────────────────
    qty_matches = {}  # position → (value, unit)
    for m in UNIT_PATTERN.finditer(lower):
        qty_matches[m.start()] = (_to_float(m.group(1)), UNIT_MAP[m.group(2).lower()])

    # ── Detect durations ──────────────────────────────────────────────────────
    duration_minutes = None
    for m in DURATION_PATTERN.finditer(lower):
        val = _to_float(m.group(1))
        unit = m.group(2).lower()
        if unit in ('jam', 'hour', 'hours', 'hr'):
            duration_minutes = (duration_minutes or 0) + val * 60
        elif unit in ('menit', 'minute', 'minutes', 'min'):
            duration_minutes = (duration_minutes or 0) + val
        elif unit in ('detik', 'second', 'seconds', 'sec'):
            duration_minutes = (duration_minutes or 0) + val / 60

    # ── Detect activities ─────────────────────────────────────────────────────
    detected_activity_spans = []
    for canonical, aliases in ACTIVITY_TOKENS.items():
        for alias in aliases:
            for m in re.finditer(re.escape(alias), lower):
                detected_activity_spans.append((m.start(), m.end(), canonical))

    # Remove overlapping spans (keep longest)
    detected_activity_spans.sort(key=lambda x: -(x[1] - x[0]))
    used_ranges = []
    final_activities = []
    for span in detected_activity_spans:
        start, end, name = span
        if not any(s <= start < e or s < end <= e for s, e in used_ranges):
            used_ranges.append((start, end))
            final_activities.append({"name": name, "duration_minutes": duration_minutes})

    result["activities"] = final_activities

    # ── Detect food items ─────────────────────────────────────────────────────
    food_spans = []
    for food in FOOD_TOKENS:
        for m in re.finditer(re.escape(food), lower):
            food_spans.append((m.start(), m.end(), food))

    food_spans.sort(key=lambda x: -(x[1] - x[0]))
    used_food_ranges = []
    for span in food_spans:
        start, end, name = span
        if not any(s <= start < e or s < end <= e for s, e in used_food_ranges):
            used_food_ranges.append((start, end))
            # Try to find a quantity immediately before/after this food token
            qty, unit = None, None
            for qpos, (qval, qunit) in qty_matches.items():
                if abs(qpos - start) < 30:  # within 30 chars
                    qty, unit = qval, qunit
                    break
            result["foods"].append({"name": name, "qty": qty, "unit": unit})

    # ── Collect standalone numbers ────────────────────────────────────────────
    result["numbers"] = [_to_float(m.group(1)) for m in NUMBER_PATTERN.finditer(lower)]

    return result


def build_nlp_context_block(parsed: dict) -> str:
    """Convert parsed NLP result into a human-readable context block for the LLM."""
    lines = []

    if parsed["foods"]:
        food_strs = []
        for f in parsed["foods"]:
            s = f["name"]
            if f["qty"]:
                s += f" ({f['qty']} {f['unit'] or ''})"
            food_strs.append(s)
        lines.append("🍽️ Makanan terdeteksi: " + ", ".join(food_strs))

    if parsed["activities"]:
        act_strs = []
        for a in parsed["activities"]:
            s = a["name"]
            if a["duration_minutes"]:
                s += f" ({a['duration_minutes']:.0f} menit)"
            act_strs.append(s)
        lines.append("🏃 Aktivitas terdeteksi: " + ", ".join(act_strs))

    return "\n".join(lines) if lines else ""


# ─────────────────────────────────────────────────────────────────────────────
#  CHATBOT CLASS
# ─────────────────────────────────────────────────────────────────────────────

class FitnessChatbot:
    def __init__(self, user_data=None):
        self.user_data = user_data
        self.client = None

        api_key = None
        if os.getenv('GROQ_API_KEY'):
            api_key = os.getenv('GROQ_API_KEY')
        elif hasattr(st, 'secrets') and 'GROQ_API_KEY' in st.secrets:
            api_key = st.secrets['GROQ_API_KEY']

        if api_key:
            self.client = Groq(api_key=api_key)

    def get_response(self, user_message, context=None, history=None):
        """Generate response using Groq API or fallback rule-based."""
        if self.client:
            return self._get_groq_response(user_message, context, history)
        else:
            return self._get_rule_based_response(user_message, context)

    def _get_groq_response(self, user_message, context, history):
        """Get response from Groq API with full conversation memory + NLP pre-processing."""
        try:
            # ── NLP tokenizer: parse user message ────────────────────────────
            parsed = parse_user_input(user_message)
            nlp_block = build_nlp_context_block(parsed)

            # ── Build system prompt ───────────────────────────────────────────
            system_prompt = self._build_system_prompt()

            if context and context.get('calories_in') is not None:
                system_prompt += (
                    f"\n\n📊 Statistik hari ini: konsumsi {context['calories_in']:.0f} kcal, "
                    f"terbakar {context['calories_out']:.0f} kcal. "
                    f"Net: {context['calories_in'] - context['calories_out']:.0f} kcal."
                )

            # Inject NLP hints into system prompt
            if nlp_block:
                system_prompt += (
                    f"\n\n[NLP PRE-PARSE — gunakan ini sebagai referensi saat memproses pesan user]\n"
                    f"{nlp_block}\n"
                    "Jika user meminta log, pastikan SEMUA item di atas (makanan DAN aktivitas) "
                    "dicatat bersama-sama dalam satu respons konfirmasi."
                )

            # ── Prepare messages ──────────────────────────────────────────────
            messages = [{"role": "system", "content": system_prompt}]

            if history and len(history) > 0:
                for msg in history[-20:]:
                    if msg['role'] in ['user', 'assistant']:
                        messages.append({"role": msg['role'], "content": msg['content']})

            messages.append({"role": "user", "content": user_message})

            # ── Tool definitions ──────────────────────────────────────────────
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "log_food",
                        "description": (
                            "Catat makanan ke food log. HANYA panggil setelah user "
                            "secara eksplisit mengkonfirmasi (misal: 'Ya', 'Catat', 'Oke'). "
                            "Jika user menyebut MAKANAN dan AKTIVITAS sekaligus dan mengkonfirmasi, "
                            "panggil tool ini DAN log_activity dalam satu respons."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "food_name": {"type": "string", "description": "Nama makanan"},
                                "calories":  {"type": "number", "description": "Total kalori (kcal)"},
                                "protein":   {"type": "number", "description": "Protein (g)"},
                                "carbs":     {"type": "number", "description": "Karbohidrat (g)"},
                                "fat":       {"type": "number", "description": "Lemak (g)"},
                                "meal_type": {
                                    "type": "string",
                                    "enum": ["Breakfast", "Lunch", "Dinner", "Snack"],
                                    "description": "Jenis makan"
                                }
                            },
                            "required": ["food_name", "calories", "protein", "carbs", "fat", "meal_type"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "log_activity",
                        "description": (
                            "Catat aktivitas/olahraga ke activity log. HANYA panggil setelah user "
                            "secara eksplisit mengkonfirmasi. "
                            "Jika user menyebut MAKANAN dan AKTIVITAS sekaligus dan mengkonfirmasi, "
                            "panggil tool ini DAN log_food dalam satu respons."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "activity_type":    {"type": "string", "description": "Nama aktivitas"},
                                "duration_minutes": {"type": "number", "description": "Durasi (menit)"},
                                "calories_burned":  {"type": "number", "description": "Kalori terbakar (kcal)"},
                                "intensity":        {
                                    "type": "string",
                                    "enum": ["Low", "Medium", "High"],
                                    "description": "Intensitas"
                                }
                            },
                            "required": ["activity_type", "duration_minutes", "calories_burned", "intensity"]
                        }
                    }
                }
            ]

            # ── First API call ────────────────────────────────────────────────
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.65,
                max_tokens=500,
                top_p=0.9,
                frequency_penalty=0.3,
                presence_penalty=0.3,
                tools=tools,
                tool_choice="auto"
            )

            response_message = response.choices[0].message

            # ── Handle tool calls (supports MULTIPLE in one response) ─────────
            if response_message.tool_calls:
                import json
                from datetime import date
                from database import add_food_log, add_activity_log

                user_id = self.user_data.get('user_id') if self.user_data else None
                tool_results = []

                # Append the assistant's tool-call message first
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in response_message.tool_calls
                    ]
                })

                # ── Process EVERY tool call ───────────────────────────────────
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    try:
                        function_args = json.loads(tool_call.function.arguments)
                    except Exception:
                        function_args = {}

                    result_text = "Data tidak tercatat (user_id tidak ditemukan)."

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
                            result_text = (
                                f"SUCCESS: Berhasil mencatat makanan '{function_args.get('food_name')}' "
                                f"dengan {function_args.get('calories', 0):.0f} kcal."
                            )

                        elif function_name == "log_activity":
                            add_activity_log(
                                user_id=user_id,
                                activity_type=function_args.get("activity_type", "Exercise"),
                                duration_minutes=function_args.get("duration_minutes", 0),
                                calories_burned=function_args.get("calories_burned", 0),
                                intensity=function_args.get("intensity", "Medium"),
                                log_date=date.today().isoformat()
                            )
                            result_text = (
                                f"SUCCESS: Berhasil mencatat aktivitas '{function_args.get('activity_type')}' "
                                f"selama {function_args.get('duration_minutes', 0):.0f} menit, "
                                f"membakar {function_args.get('calories_burned', 0):.0f} kcal."
                            )

                    # Append tool result
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": result_text
                    })
                    tool_results.append(result_text)

                # ── Second API call: conversational wrap-up ───────────────────
                second_response = self.client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    temperature=0.65,
                    max_tokens=400,
                    top_p=0.9
                )
                return self._clean_response(second_response.choices[0].message.content)

            return self._clean_response(response_message.content)

        except Exception as e:
            return self._handle_api_error(e)

    # ─────────────────────────────────────────────────────────────────────────
    #  HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _handle_api_error(self, e):
        error_str = str(e).lower()

        if '429' in str(e) or 'rate_limit' in error_str or 'tokens per day' in error_str:
            import re as _re
            wait_match = _re.search(
                r'try again in (\d+m\d+\.?\d*s|\d+\.?\d*s|\d+ minute)', str(e), _re.IGNORECASE
            )
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

        if '401' in str(e) or 'invalid api key' in error_str or 'unauthorized' in error_str:
            return (
                "🔑 **FitBot tidak dapat terhubung saat ini.**\n\n"
                "Terjadi masalah konfigurasi pada layanan AI. "
                "Silakan hubungi admin aplikasi."
            )

        if 'timeout' in error_str or 'connection' in error_str or 'network' in error_str:
            return (
                "🌐 **Koneksi bermasalah.**\n\n"
                "FitBot tidak dapat terhubung ke server AI saat ini. "
                "Pastikan koneksi internet kamu stabil, lalu coba lagi."
            )

        if '503' in str(e) or '500' in str(e) or 'overloaded' in error_str:
            return (
                "🛠️ **Server AI sedang sibuk.**\n\n"
                "Terlalu banyak pengguna mengakses FitBot sekarang. "
                "Tunggu sebentar dan coba lagi ya! 🙏"
            )

        return (
            "😅 **FitBot mengalami kendala teknis.**\n\n"
            "Maaf atas ketidaknyamanannya! Silakan coba kirim pesan lagi. "
            "Jika masalah terus berlanjut, coba refresh halaman."
        )

    def _clean_response(self, text):
        if text is None:
            return ""
        cleaned = re.sub(r'<function=\w+>.*?</function>', '', text, flags=re.DOTALL)
        cleaned = re.sub(r'```json\s*\{[^`]*\}\s*```', '', cleaned, flags=re.DOTALL)
        cleaned = cleaned.strip()
        if not cleaned:
            cleaned = "✅ Sudah dicatat ke log Anda!"
        return cleaned

    def _build_system_prompt(self):
        if not self.user_data:
            return self._get_default_system_prompt()

        height_m = self.user_data.get('height_cm', 170) / 100
        weight   = self.user_data.get('weight_kg', 70)
        bmi      = weight / (height_m ** 2) if height_m > 0 else 22
        bmi_cat  = ("Underweight" if bmi < 18.5 else
                    "Normal"      if bmi < 25   else
                    "Overweight"  if bmi < 30   else "Obese")

        prompt = f"""Kamu adalah FitBot, AI coach kebugaran dan nutrisi yang ahli, empatik, dan berbasis bukti ilmiah.

PROFIL USER:
- Nama: {self.user_data.get('name', 'User')}
- Usia: {self.user_data.get('age', 'N/A')} tahun
- Jenis Kelamin: {self.user_data.get('gender', 'N/A')}
- Tinggi: {self.user_data.get('height_cm', 'N/A')} cm | Berat: {self.user_data.get('weight_kg', 'N/A')} kg
- BMI: {bmi:.1f} ({bmi_cat})
- BMR: {self.user_data.get('bmr', 0):.0f} kcal/hari | TDEE: {self.user_data.get('tdee', 0):.0f} kcal/hari
- Target Kalori: {self.user_data.get('daily_target_calories', 0):.0f} kcal/hari
- Tujuan: {self.user_data.get('fitness_goal', 'Maintain Weight')}

ATURAN RESPONS (WAJIB DIIKUTI):

1. **BAHASA**: Balas dalam bahasa yang sama dengan user. Indonesia → Indonesia.

2. **KONFIRMASI LOG (SANGAT PENTING)**:
   - Saat user menceritakan makanan DAN/ATAU aktivitas yang sudah dilakukan, TANYA DULU apakah ingin dicatat.
   - Dalam satu pertanyaan konfirmasi, sebutkan SEMUA item yang akan dicatat (makanan + aktivitas sekaligus).
   - Contoh: "Saya akan mencatat **popcorn 100g** (~380 kcal) dan **jalan kaki 20 menit** (~80 kcal terbakar). Mau saya catat keduanya?"
   - Setelah user konfirmasi ('Ya', 'Oke', 'Catat', 'Boleh', 'Iya', 'Yep', dll.), PANGGIL SEMUA tool yang relevan sekaligus — jangan hanya satu.

3. **MULTIPLE TOOL CALLS**: Jika ada makanan DAN aktivitas yang perlu dicatat, panggil `log_food` DAN `log_activity` dalam satu giliran respons. JANGAN hanya memanggil satu saja.

4. **NLP HINTS**: Sistem telah menganalisis pesan user dan menyediakan [NLP PRE-PARSE] di atas. Gunakan info tersebut untuk memastikan tidak ada makanan atau aktivitas yang terlewat.

5. **KONKRET**: Berikan estimasi kalori/makro yang realistis berdasarkan database nutrisi (contoh: popcorn 100g ≈ 375-400 kcal).

6. **RINGKAS**: Respons 3-5 kalimat + poin-poin jika perlu. Gunakan emoji secukupnya (💪, 🥗, 🏃).

7. **FOKUS**: Tetap pada topik kesehatan dan kebugaran. Jika ditanya topik lain, arahkan kembali ke fitness.

8. **JANGAN PERNAH** menulis sintaks function call mentah seperti `<function=log_food>{{...}}</function>` di teks respons.

Ingat: Pengguna adalah manusia nyata yang ingin sehat. Buat setiap respons berarti!"""

        return prompt

    def _get_default_system_prompt(self):
        return """Kamu adalah FitBot, coach kebugaran dan nutrisi yang ramah dan berbasis sains.
Berikan saran praktis tentang olahraga, diet, penurunan berat badan, dan gaya hidup sehat.
Jawaban ringkas (maks 200 kata) dan dapat ditindaklanjuti. Gunakan emoji agar menarik.
Balas dalam bahasa yang sama dengan user (Indonesia atau Inggris).

ATURAN PENTING:
- Jika user menyebut makanan DAN aktivitas sekaligus, konfirmasi KEDUANYA dalam satu pertanyaan.
- Setelah user konfirmasi, panggil log_food DAN log_activity bersamaan.
- Jangan panggil tool tanpa konfirmasi eksplisit dari user."""

    def _get_rule_based_response(self, user_message, context):
        """Enhanced fallback when API is not available."""
        msg = user_message.lower().strip()

        # Run NLP parser even in rule-based mode for better detection
        parsed = parse_user_input(msg)

        if parsed["foods"] or parsed["activities"]:
            items = []
            for f in parsed["foods"]:
                s = f["name"]
                if f["qty"]:
                    s += f" {f['qty']} {f['unit'] or ''}"
                items.append(f"🍽️ {s}")
            for a in parsed["activities"]:
                s = a["name"]
                if a["duration_minutes"]:
                    s += f" {a['duration_minutes']:.0f} menit"
                items.append(f"🏃 {s}")
            item_list = "\n".join(items)
            return (
                f"Saya mendeteksi:\n{item_list}\n\n"
                "Apakah kamu ingin saya catat semua data ini ke log? "
                "(FitBot sedang dalam mode offline — untuk pencatatan otomatis, hubungkan API Key)"
            )

        if any(w in msg for w in ['lose weight', 'weight loss', 'turun berat', 'diet turun']):
            if self.user_data:
                deficit = self.user_data.get('tdee', 2500) - self.user_data.get('daily_target_calories', 2000)
                return (
                    f"📉 **Strategi Turun Berat Badan**\n\n"
                    f"TDEE kamu {self.user_data.get('tdee', 0):.0f} kcal, target {self.user_data.get('daily_target_calories', 0):.0f} kcal "
                    f"(defisit {deficit:.0f} kcal/hari).\n\n"
                    "✅ **Langkah utama:**\n"
                    f"- Makan {self.user_data.get('daily_target_calories', 0):.0f} kcal/hari\n"
                    f"- Protein {self.user_data.get('weight_kg', 70)*1.8:.0f}g/hari\n"
                    "- Latihan kekuatan 3x/minggu + 150 menit kardio\n"
                    "- Tidur 7-8 jam\n\nMau contoh menu makan atau program latihan?"
                )

        if any(w in msg for w in ['gain muscle', 'build muscle', 'tambah otot', 'bulk']):
            if self.user_data:
                return (
                    f"💪 **Program Tambah Otot**\n\n"
                    f"Maintenance kamu {self.user_data.get('tdee', 0):.0f} kcal.\n\n"
                    "✅ **Essentials:**\n"
                    f"- Kalori: {self.user_data.get('daily_target_calories', 0):.0f} kcal\n"
                    f"- Protein: {self.user_data.get('weight_kg', 70)*1.8:.0f}g/hari\n"
                    "- Progressive overload\n"
                    "- Compound lifts: squat, deadlift, bench, row\n\nMau program 3 hari?"
                )

        if any(w in msg for w in ['kalori', 'calorie', 'calories']):
            if context and context.get('calories_in') is not None:
                remaining = (self.user_data.get('daily_target_calories', 2000) - context['calories_in']) if self.user_data else 0
                return (
                    f"🍽️ **Status Kalori Hari Ini**\n"
                    f"- Konsumsi: {context['calories_in']:.0f} kcal\n"
                    f"- Terbakar: {context['calories_out']:.0f} kcal\n"
                    f"- Net: {context['calories_in'] - context['calories_out']:.0f} kcal\n"
                    f"- Target: {self.user_data.get('daily_target_calories', 'N/A') if self.user_data else 'N/A'} kcal\n"
                    f"- Sisa: {remaining:.0f} kcal"
                )

        if any(w in msg for w in ['workout', 'exercise', 'olahraga', 'gym', 'latihan']):
            return (
                "🏋️ **Rekomendasi Latihan**\n"
                "- 3x/minggu full-body strength training\n"
                "- 150 menit kardio moderat/minggu\n"
                "- HIIT 1x/minggu\n"
                "- Target langkah harian: 8.000+\n\nMau program spesifik?"
            )

        if any(w in msg for w in ['hello', 'hi', 'halo', 'hai', 'selamat']):
            hour = datetime.now().hour
            greeting = "Selamat pagi! ☀️" if hour < 12 else ("Selamat siang! 🌤️" if hour < 18 else "Selamat malam! 🌙")
            return (
                f"{greeting} Saya FitBot, coach kebugaran pribadi kamu!\n\n"
                "Saya bisa membantu:\n"
                "• 🥗 Nutrisi & perencanaan makan\n"
                "• 💪 Program latihan\n"
                "• 🔥 Hitung kalori & makro\n"
                "• 📈 Motivasi & progres\n\nApa yang ingin kamu tanyakan?"
            )

        return (
            "Halo! Saya siap membantu perjalanan kebugaranmu. 😊\n\n"
            "Coba tanyakan:\n"
            "- \"Saya habis makan nasi goreng 1 porsi, catat ya!\"\n"
            "- \"Saya lari 30 menit tadi, tolong catat\"\n"
            "- \"Berapa kalori yang harus saya makan?\"\n"
            "- \"Beri saya program latihan\""
        )

    def clear_history(self):
        self.conversation_history = []