"""
=============================================================
FUTO STUDENT CHATBOT — Backend  (futo_chatbot.py)
=============================================================
University : Federal University of Technology, Owerri (FUTO)
Project    : AI-Powered Student Chat Service
Version    : 2.0
Python     : 3.13 / 3.14 compatible

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AI PIPELINE (3-layer hybrid):
  Layer 1 — Rule-Based NLP (Regex keyword matching)
             Highest priority; covers all FUTO-specific topics
  Layer 2 — Supervised ML (TF-IDF + Logistic Regression)
             Pre-trained on CLINC150 (~15k samples, 150 intents)
             Loaded from futo_model.pkl — NO training at startup
  Layer 3 — MyMemory AI Fallback (External free REST API)
             When layers 1 & 2 return "unknown", a free HTTP
             call is made to the MyMemory Translation Memory API
             which acts as a general-knowledge Q&A fallback.

AI TYPES EXPLAINED:
  TF-IDF (Term Frequency-Inverse Document Frequency)
    Converts raw text into a numeric sparse matrix. High score =
    word is frequent in THIS doc but rare elsewhere. This is the
    FEATURE EXTRACTOR fed into Logistic Regression.

  Logistic Regression (Multinomial)
    A probabilistic linear classifier. For each incoming text
    vector it outputs P(intent|text) across all 150 classes and
    picks the highest. Fast, interpretable, and very competitive
    for short-text intent classification.

  Rule-Based NLP (Regex)
    Deterministic pattern matching using Python re module.
    Used because FUTO vocabulary (remita, RRR, CGPA, Ihiagwa,
    SEET, portal.futo.edu.ng) will NEVER appear in CLINC150
    training data; regex guarantees their detection.

  MyMemory API Fallback
    MyMemory is a free translation memory REST service.
    Querying it with langpair=en|en performs a semantic memory
    lookup that approximates a general knowledge Q&A.
    Zero cost, no API key required, 10k chars/day free tier.

MODULES (all support Python 3.13+):
  flask, flask-cors, scikit-learn, requests
  (pandas NOT needed at runtime — only used in train_model.py)
=============================================================
"""

# ── Standard library ─────────────────────────────────────────
import re
import os
import pickle
import random
import logging
from datetime import datetime
from collections import defaultdict

# ── Third-party ──────────────────────────────────────────────
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import warnings
warnings.filterwarnings("ignore")

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("FUTObot")


# =============================================================
# SECTION 1 — STUDENT IDENTITY
# =============================================================
STUDENT_PROFILE = {
    "full_name"  : "ANYADIKE SHARON NMESOMACHUKWU",
    "reg_number" : "20231364542",
    "department" : "CYBERSECURITY",
    "level"      : "300 Level",
    "session"    : "2025/2026",
    "faculty"    : "SICT",
}


# =============================================================
# SECTION 2 — FUTO KNOWLEDGE BASE
# =============================================================
FUTO_KB = {

    "course_registration": [
        (
            "📚 **Course Registration at FUTO:**\n\n"
            "• Portal: **portal.futo.edu.ng** → Course Registration\n"
            "• Window: opens **2 weeks** after session resumption\n"
            "• Max load: **24 credit units** | Min load: **15 credit units**\n"
            "• Late penalty: **₦5,000** — avoid it!\n\n"
            "**Steps:**\n"
            "1. Login with matric number & password\n"
            "2. Select current session & semester\n"
            "3. Add required + elective courses\n"
            "4. Submit → download & print your slip\n\n"
            "💡 *Confirm your course list with your Level Adviser first.*"
        ),
        (
            "📋 **How to Register Courses (FUTO):**\n\n"
            "→ portal.futo.edu.ng → **Academics → Course Registration**\n"
            "→ Electives need adviser pre-approval\n"
            "→ Max: **24 units** per semester\n\n"
            "⚠ *Unregistered courses = no result. Don't skip this step!*"
        ),
    ],

    "school_fees": [
        (
            "💰 **School Fees — FUTO 2024/2025:**\n\n"
            "| Level         | Approx. Fee |\n"
            "|---------------|-------------|\n"
            "| 100 Level     | ₦56,000     |\n"
            "| 200 Level     | ₦50,000     |\n"
            "| 300–500 Level | ₦46,000     |\n"
            "| Postgraduate  | ₦80,000+    |\n\n"
            "**Payment Steps:**\n"
            "1. Portal → **Finance → Generate Invoice**\n"
            "2. Copy your **RRR** (Remita Retrieval Reference)\n"
            "3. Pay via remita.net or any commercial bank\n"
            "4. Upload receipt on portal within **48 hours**\n\n"
            "⚠ *Always verify fees on portal.futo.edu.ng — subject to change.*"
        ),
        (
            "🏦 **Paying FUTO School Fees (Remita):**\n\n"
            "• Generate invoice on the student portal\n"
            "• Pay using your **RRR number** at any bank\n"
            "• Online: remita.net → Pay Federal Government Agency → FUTO\n\n"
            "📞 Bursary: Admin Block, Ground Floor\n"
            "📧 bursary@futo.edu.ng"
        ),
    ],

    "timetable": [
        (
            "🗓️ **FUTO Class Timetable:**\n\n"
            "• Portal → **Academics → Timetable → Select Semester**\n"
            "• Also on departmental notice boards each semester\n"
            "• Lecture hours: **8:00 AM – 6:00 PM** (Mon–Fri)\n"
            "• Each slot = **1 hr** | Lab sessions = **3 hrs**\n"
            "• Exam timetable released **3 weeks** before exams\n\n"
            "💡 *Your course rep usually shares it on WhatsApp first!*"
        ),
        (
            "📅 **Finding Your Timetable at FUTO:**\n\n"
            "→ Student Portal → Academics → Timetable\n"
            "→ Departmental notice boards (faculty ground floor)\n"
            "→ Department WhatsApp group\n\n"
            "⚠ *Confirm lecture venues — some classes shift to bigger halls.*"
        ),
    ],

    "hostel": [
        (
            "🏠 **FUTO Hostel & Accommodation:**\n\n"
            "**On-Campus Halls:**\n"
            "• Male: Python Hall, Java Hall, Hall A/B/C\n"
            "• Female: Female Hostel Block A–D\n"
            "• Rooms: shared (4–6 students per room)\n\n"
            "**Application:**\n"
            "1. Portal → **Accommodation → Apply for Hostel**\n"
            "2. Pay ~**₦25,000/session** via Remita\n"
            "3. Allocation result within **1 week**\n\n"
            "🏘️ *Off-campus: Ihiagwa, Obinze, Nekede — popular student areas.*"
        ),
        (
            "🛏️ **Hostel Allocation Tips (FUTO):**\n\n"
            "• Priority: **100-level & final-year** students\n"
            "• Fee: ~₦25,000/session (covers both semesters)\n"
            "• No hostel? Visit Students Affairs Unit — Admin Block\n\n"
            "📞 Students Affairs: 083-230-0101"
        ),
    ],

    "results": [
        (
            "📊 **Checking Results (FUTO):**\n\n"
            "→ portal.futo.edu.ng → **Academics → Results → Select Semester**\n\n"
            "**FUTO GPA Scale:**\n"
            "| Grade | Score   | Points |\n"
            "|-------|---------|--------|\n"
            "| A     | 70–100  | 5.0    |\n"
            "| B     | 60–69   | 4.0    |\n"
            "| C     | 50–59   | 3.0    |\n"
            "| D     | 45–49   | 2.0    |\n"
            "| E     | 40–44   | 1.0    |\n"
            "| F     | 0–39    | 0.0    |\n\n"
            "🎯 First Class: CGPA ≥ **4.50** | Minimum to graduate: **1.0**\n\n"
            "⚠ *Missing result? See your department Exam Officer immediately.*"
        ),
        (
            "🎓 **FUTO Results & Honours:**\n\n"
            "• Released after Senate ratification (6–8 weeks post-exam)\n\n"
            "| Class         | CGPA        |\n"
            "|---------------|-------------|\n"
            "| First Class   | 4.50 – 5.00 |\n"
            "| Second Upper  | 3.50 – 4.49 |\n"
            "| Second Lower  | 2.40 – 3.49 |\n"
            "| Third Class   | 1.50 – 2.39 |\n"
            "| Pass          | 1.00 – 1.49 |"
        ),
    ],

    "gpa_info": [
        (
            "📐 **How GPA is Calculated at FUTO:**\n\n"
            "**Formula:** GPA = Σ(Grade Points × Credit Units) ÷ Total Credit Units\n\n"
            "**Example:**\n"
            "| Course    | Units  | Grade | GP  | Total  |\n"
            "|-----------|--------|-------|-----|--------|\n"
            "| MTH 201   | 3      | A     | 5.0 | 15.0   |\n"
            "| PHY 202   | 3      | B     | 4.0 | 12.0   |\n"
            "| EEE 301   | 4      | C     | 3.0 | 12.0   |\n"
            "| **Total** | **10** |       |     | **39.0** |\n\n"
            "GPA = 39.0 ÷ 10 = **3.90** (Second Upper Class)\n\n"
            "💡 *High-credit courses affect your GPA more — prioritise them!*"
        ),
    ],

    "department_contacts": [
        (
            "📞 **FUTO Department Contacts:**\n\n"
            "| Department            | Location      | Email              |\n"
            "|-----------------------|---------------|--------------------|\n"
            "| Computer Science      | SICT, Rm 201  | cs@futo.edu.ng     |\n"
            "| Electronic & Comp Eng | SEET, Rm 105  | ece@futo.edu.ng    |\n"
            "| Electrical Engineering| SEET, Rm 103  | ee@futo.edu.ng     |\n"
            "| Civil Engineering     | SCET Block    | civil@futo.edu.ng  |\n"
            "| Mechanical Eng        | SMEET Block   | mech@futo.edu.ng   |\n"
            "| Architecture          | SAAT Block    | arch@futo.edu.ng   |\n\n"
            "🏛️ General: **info@futo.edu.ng** | ☎ 083-230-0101"
        ),
    ],

    "library": [
        (
            "📚 **FUTO Library:**\n\n"
            "• **Location:** Central Library, beside Senate Building\n"
            "• **Hours:** Mon–Fri 8 AM – 9 PM | Sat 9 AM – 5 PM\n"
            "• **E-Library:** elibrary.futo.edu.ng (portal login)\n"
            "• **Library card:** ₦2,000 (one-time fee)\n\n"
            "**Research Databases:** JSTOR | IEEE Xplore | Springer | HINARI\n\n"
            "📧 library@futo.edu.ng"
        ),
    ],

    "clearance": [
        (
            "🔖 **Student Clearance Process (FUTO):**\n\n"
            "Required at the **start of each session** before course registration.\n\n"
            "**Visit these offices in order:**\n"
            "1. 🏦 Bursary — confirm fee payment\n"
            "2. 📚 Library — return all borrowed books\n"
            "3. 🏥 Medical Centre — health screening\n"
            "4. 🔒 Security Unit\n"
            "5. 🎓 Dean of Students\n"
            "6. 🏫 Your Department\n\n"
            "⚠ *No clearance = No course registration = No exams!*"
        ),
    ],

    "exam_rules": [
        (
            "✏️ **FUTO Examination Rules:**\n\n"
            "**Before the exam:**\n"
            "• Download **Exam Card** from portal → Academics\n"
            "• Arrive **30 minutes** early\n"
            "• Bring your **Student ID** — no ID, no entry!\n\n"
            "**In the hall:**\n"
            "• No phones, smartwatches, or Bluetooth devices\n"
            "• Non-programmable calculators only\n"
            "• Water in clear bottles is allowed\n\n"
            "**Malpractice penalties:**\n"
            "• 1st offence: ZERO in that paper\n"
            "• Repeat: suspension or expulsion\n\n"
            "📌 Exam Card: portal → **Academics → Exam Card**"
        ),
    ],

    "transcript": [
        (
            "📄 **Academic Transcript (FUTO):**\n\n"
            "• Apply at: **Academic Registry**, Admin Block\n"
            "• Fee: **₦15,000** local | **₦25,000** international\n"
            "• Processing: **2–4 weeks**\n\n"
            "**Required:**\n"
            "→ Student ID copy | Application letter | Fee receipt\n\n"
            "📧 registry@futo.edu.ng\n"
            "💡 *Postgrad admission? Request direct Registry-to-institution dispatch.*"
        ),
    ],

    "admission": [
        (
            "🎓 **FUTO Admission Requirements:**\n\n"
            "**UTME (100 Level):**\n"
            "• Min JAMB score: **180** (Science) | **160** (Arts)\n"
            "• 5 O'Level credits including Math & English\n"
            "• Compulsory Post-UTME screening\n\n"
            "**Direct Entry (200 Level):**\n"
            "• ND/HND Lower Credit min, or 2 A-Level passes\n\n"
            "📅 Screening dates: futo.edu.ng\n"
            "📧 admissions@futo.edu.ng"
        ),
    ],

    "scholarship": [
        (
            "🏆 **Scholarships for FUTO Students:**\n\n"
            "| Scholarship      | Eligibility           | Value         |\n"
            "|------------------|-----------------------|---------------|\n"
            "| FGN Scholarship  | CGPA ≥ 3.5, 200L+    | ₦60,000/yr    |\n"
            "| NNPC/Total       | Engineering students  | Varies        |\n"
            "| NDDC             | Niger Delta origin    | ₦100,000/yr   |\n"
            "| FUTO Merit Award | Top 5 per dept        | Cash + cert   |\n\n"
            "**Apply:** Portal → **Scholarships & Awards**\n"
            "📧 scholar@futo.edu.ng"
        ),
    ],

    "portal_help": [
        (
            "💻 **FUTO Portal Troubleshooting:**\n\n"
            "**Can't login?**\n"
            "• Default password: your **date of birth** (DDMMYYYY)\n"
            "• Click 'Forgot Password' on portal.futo.edu.ng\n"
            "• Password reset: ICT Directorate, SICT Building\n\n"
            "**Portal slow/down?**\n"
            "• Best time: **6–9 AM** (low traffic)\n"
            "• Use Chrome; clear browser cache\n\n"
            "📞 ICT Helpdesk: ict@futo.edu.ng | 083-230-0102"
        ),
    ],

    "general_greeting": [
        (
            "👋 **Hello! Welcome to FUTO Student Services!**\n\n"
            "I'm **FUTObot**, your AI academic assistant.\n\n"
            "I can help with:\n"
            "📚 Registration  •  💰 Fees  •  🗓️ Timetable\n"
            "🏠 Hostel  •  📊 Results  •  ✏️ Exam rules\n"
            "📄 Transcript  •  🏆 Scholarships  •  📞 Contacts\n"
            "💻 Portal help  •  📐 GPA calculation\n\n"
            "What can I help you with today?"
        ),
        "Hello! 😊 How can I help you today? Ask me anything about FUTO!",
        "Hi there! 👋 I'm FUTObot — your 24/7 FUTO assistant. What do you need?",
    ],

    "farewell": [
        "Goodbye! 👋 Best of luck with your studies. Come back anytime! 🎓",
        "Take care! Every challenge is a stepping stone to greatness. #FUTOPride 💚",
        "See you! Don't hesitate to return if you need help. 😊",
    ],

    "thanks": [
        "You're welcome! 😊 Is there anything else I can help you with?",
        "Happy to help! That's what I'm here for. 🤖",
        "Anytime! Feel free to ask more questions. 💚",
    ],

    "bot_identity": [
        (
            "🤖 **About FUTObot:**\n\n"
            "I am an **AI-powered student assistant** for FUTO.\n\n"
            "**AI Architecture:**\n"
            "• Layer 1: Rule-Based NLP (Regex pattern matching)\n"
            "• Layer 2: TF-IDF + Logistic Regression ML model\n"
            "  pre-trained on CLINC150 (15,000+ intent samples)\n"
            "• Layer 3: MyMemory AI API fallback (free REST API)\n\n"
            "**Stack:** Python 3.13 · Flask · scikit-learn · Bootstrap 5\n"
            "**Purpose:** 24/7 FUTO student support"
        ),
    ],

    "unknown": [
        (
            "🤔 I'm not sure I understand that. Could you rephrase?\n\n"
            "**I can help with:**\n"
            "→ Course registration, school fees, timetable\n"
            "→ Hostel, results, GPA, scholarships\n"
            "→ Exam rules, clearance, transcript, portal\n\n"
            "*Searching for an answer via AI fallback…*"
        ),
        (
            "😕 That's outside my FUTO knowledge base. Try:\n"
            "→ 'How do I register my courses?'\n"
            "→ 'What are the hostel fees?'\n"
            "→ 'How do I check my results?'"
        ),
    ],
}


# =============================================================
# SECTION 3 — KEYWORD RULES  (Layer 1)
# =============================================================
FUTO_KEYWORD_RULES = [
    (r"course.?regist|register.?course|add course|portal.*course|credit unit|matric.*course|course.*load",  "course_registration"),
    (r"school fee|tuition|pay fee|remita|payment|bursary|rrr\b|invoice|fee.*pay|pay.*fee",                  "school_fees"),
    (r"timetable|class schedule|lecture time|lecture hour|exam timetable|class.*time|when.*lecture",         "timetable"),
    (r"hostel|accommodation|\bhall\b|dormitory|where.?stay|on.?campus.*live|bedspace",                      "hostel"),
    (r"result|gpa|cgpa|grade.*point|semester result|check.?result|academic result",                         "results"),
    (r"gpa.*calculat|calculat.*gpa|how.*gpa.*work|grade.*formula|credit.*unit.*formula",                    "gpa_info"),
    (r"department contact|hod|head of department|faculty.*email|email.*dept|staff.*contact",                 "department_contacts"),
    (r"library|e.?library|borrow.*book|ieee|jstor|research.*database|library.*hour",                        "library"),
    (r"clearance|clear form|sign.?off|clearance.*form|medical.*clear|library clearance",                    "clearance"),
    (r"exam.?rule|malpractice|exam card|examination rule|invigil|hall.*rule",                               "exam_rules"),
    (r"transcript|academic record|registry|official.*result.*document|request.*transcript",                 "transcript"),
    (r"admission|jamb|post.?utme|direct entry|screening.*futo|futo.*admission",                             "admission"),
    (r"scholarship|bursary.*award|merit award|nnpc|nddc|fgn.*scholar|student.*award",                      "scholarship"),
    (r"portal.*login|cant.*login|portal.*down|password.*reset|portal.*problem|ict.*help",                  "portal_help"),
    (r"who are you|what are you|your name|tell.*about yourself|are you (a )?bot|futobot",                  "bot_identity"),
    (r"\bhi\b|\bhello\b|good morning|good afternoon|good evening|\bhey\b|howdy|greetings",                  "general_greeting"),
    (r"\bbye\b|goodbye|see you|farewell|take care|\bexit\b|\bquit\b",                                       "farewell"),
    (r"\bthank|\bthanks\b|\bappreciate\b|grateful|well done",                                               "thanks"),
]

FUTO_RULES_COMPILED = [
    (re.compile(pat, re.IGNORECASE), intent)
    for pat, intent in FUTO_KEYWORD_RULES
]


# =============================================================
# SECTION 4 — MYMEMORY AI FALLBACK  (Layer 3)
# =============================================================
MYMEMORY_EMAIL = "futobot@futo.edu.ng"
MYMEMORY_CACHE: dict = {}


def query_mymemory(user_text: str) -> "str | None":
    """
    Query the MyMemory REST API as a general-knowledge fallback.
    Returns a formatted string, or None on failure / no match.
    """
    cache_key = user_text.strip().lower()
    if cache_key in MYMEMORY_CACHE:
        log.info("MyMemory: cache hit")
        return MYMEMORY_CACHE[cache_key]

    try:
        query = f"University student asking: {user_text}"
        resp  = requests.get(
            "https://api.mymemory.translated.net/get",
            params={"q": query, "langpair": "en|en", "de": MYMEMORY_EMAIL},
            timeout=5,
        )
        resp.raise_for_status()
        data   = resp.json()
        answer = data.get("responseData", {}).get("translatedText", "").strip()

        if not answer or answer.lower() == query.lower() or len(answer) < 15:
            return None

        result = (
            f"🌐 **AI Fallback Answer (MyMemory):**\n\n{answer}\n\n"
            "💡 *For FUTO-specific queries, try keywords like "
            "'registration', 'hostel', 'results', or 'school fees'.*"
        )
        MYMEMORY_CACHE[cache_key] = result
        log.info("MyMemory: success")
        return result

    except requests.Timeout:
        log.warning("MyMemory: timed out")
        return None
    except Exception as exc:
        log.warning(f"MyMemory: error — {exc}")
        return None


# =============================================================
# SECTION 5 — ML MODEL  (Layer 2) — LOAD PRE-TRAINED
# =============================================================
def load_pretrained_model(model_path: str):
    """
    Load the pre-trained TF-IDF + Logistic Regression pipeline
    from futo_model.pkl generated by train_model.py.

    Loading a pkl uses ~80-150 MB RAM vs ~600 MB for training —
    critical for staying within Render's 512 MB free tier.

    Returns: (pipeline, label_encoder, accuracy)
    """
    log.info(f"Loading pre-trained model from {model_path} …")
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    acc = data.get("accuracy", 0.0)
    log.info(f"Model loaded! Stored accuracy: {acc:.2%}" if acc else "Model loaded!")
    return data["pipeline"], data["label_encoder"], acc


# =============================================================
# SECTION 6 — INTENT ROUTER
# =============================================================
CLINC_INTENT_TO_FUTO = {
    110: "timetable",
    118: "timetable",
    116: "course_registration",
    93 : "general_greeting",
    96 : "general_greeting",
    127: "bot_identity",
    135: "bot_identity",
    121: "bot_identity",
    123: "bot_identity",
    131: "general_greeting",
    129: "farewell",
    130: "thanks",
    132: "general_greeting",
    141: "unknown",
    150: "unknown",
    139: "unknown",
}


def classify_intent(user_text: str, pipeline, label_enc) -> str:
    """Classify intent: Layer 1 (rules) → Layer 2 (ML)."""
    cleaned = user_text.strip().lower()

    # Layer 1 — keyword rules (highest priority)
    for pattern, intent_key in FUTO_RULES_COMPILED:
        if pattern.search(cleaned):
            log.info(f"Intent (rules): {intent_key}")
            return intent_key

    # Layer 2 — ML prediction
    pred   = pipeline.predict([user_text])[0]
    label  = label_enc.inverse_transform([pred])[0]
    cid    = int(label)
    mapped = CLINC_INTENT_TO_FUTO.get(cid, "unknown")
    log.info(f"Intent (ML): CLINC {cid} → {mapped}")
    return mapped


def classify_intent_safe(text: str) -> str:
    """Safe wrapper — falls back to rules-only if model not loaded."""
    if MODEL_READY:
        return classify_intent(text, ML_PIPELINE, LABEL_ENC)
    cleaned = text.strip().lower()
    for pattern, intent_key in FUTO_RULES_COMPILED:
        if pattern.search(cleaned):
            return intent_key
    return "unknown"


def get_kb_response(intent_key: str) -> str:
    return random.choice(FUTO_KB.get(intent_key, FUTO_KB["unknown"]))


# =============================================================
# SECTION 7 — ANALYTICS STORE
# =============================================================
ANALYTICS = {
    "total_messages" : 0,
    "intent_counts"  : defaultdict(int),
    "fallback_count" : 0,
    "upvotes"        : 0,
    "downvotes"      : 0,
    "session_start"  : datetime.now().isoformat(),
}


# =============================================================
# SECTION 8 — FLASK APP
# =============================================================
app = Flask(__name__)
CORS(app)

# ── Load pre-trained model ────────────────────────────────────
MODEL_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "futo_model.pkl")
MODEL_READY = False
ML_PIPELINE = None
LABEL_ENC   = None
MODEL_ACC   = 0.0

try:
    ML_PIPELINE, LABEL_ENC, MODEL_ACC = load_pretrained_model(MODEL_PATH)
    MODEL_READY = True
    log.info("ML model ready!")
except FileNotFoundError:
    log.warning(f"'{MODEL_PATH}' not found — keyword-only + MyMemory mode.")
    log.warning("Run 'python train_model.py' locally to generate futo_model.pkl")
except Exception as exc:
    log.error(f"Model load failed: {exc}")

# ── Load HTML template ────────────────────────────────────────
_HTML_PATH    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "futo_chatbot.html")
HTML_TEMPLATE = open(_HTML_PATH, encoding="utf-8").read()


# =============================================================
# SECTION 9 — ROUTES
# =============================================================
@app.route("/")
def index():
    """Serve the chat UI with student profile injected."""
    html = HTML_TEMPLATE
    html = html.replace("{{ student.full_name }}",  STUDENT_PROFILE["full_name"])
    html = html.replace("{{ student.reg_number }}",  STUDENT_PROFILE["reg_number"])
    html = html.replace("{{ student.department }}",  STUDENT_PROFILE["department"])
    html = html.replace("{{ student.level }}",       STUDENT_PROFILE["level"])
    html = html.replace("{{ student.session }}",     STUDENT_PROFILE["session"])
    html = html.replace("{{ student.faculty }}",     STUDENT_PROFILE["faculty"])
    html = html.replace("{{ model_acc }}",           f"{MODEL_ACC:.1%}" if MODEL_ACC else "keyword-only")
    return html


@app.route("/chat", methods=["POST"])
def chat():
    """
    Chat endpoint — 3-layer pipeline.
    Request  JSON : { "message": "..." }
    Response JSON : { "reply", "intent", "source", "timestamp" }
    """
    data     = request.get_json(force=True)
    user_msg = data.get("message", "").strip()

    if not user_msg:
        return jsonify({"reply": "Please type a message.", "intent": "none",
                        "source": "none", "timestamp": ""})

    ANALYTICS["total_messages"] += 1
    intent = classify_intent_safe(user_msg)
    ANALYTICS["intent_counts"][intent] += 1

    source = "kb"
    reply  = get_kb_response(intent)

    # Layer 3 — MyMemory fallback for unknown intents
    if intent == "unknown":
        mm = query_mymemory(user_msg)
        if mm:
            reply  = mm
            source = "mymemory"
            ANALYTICS["fallback_count"] += 1
        else:
            source = "fallback"

    return jsonify({
        "reply"    : reply,
        "intent"   : intent,
        "source"   : source,
        "timestamp": datetime.now().strftime("%I:%M %p"),
    })


@app.route("/feedback", methods=["POST"])
def feedback():
    """Accept thumbs-up / thumbs-down votes from the UI."""
    data = request.get_json(force=True)
    if data.get("vote") == "up":
        ANALYTICS["upvotes"] += 1
    elif data.get("vote") == "down":
        ANALYTICS["downvotes"] += 1
    return jsonify({"status": "ok"})


@app.route("/stats")
def stats():
    """Live analytics for the sidebar dashboard."""
    top   = sorted(ANALYTICS["intent_counts"].items(), key=lambda x: x[1], reverse=True)[:5]
    start = datetime.fromisoformat(ANALYTICS["session_start"])
    secs  = int((datetime.now() - start).total_seconds())
    h, r  = divmod(secs, 3600)
    m, s  = divmod(r, 60)
    return jsonify({
        "total_messages": ANALYTICS["total_messages"],
        "fallback_count": ANALYTICS["fallback_count"],
        "upvotes"       : ANALYTICS["upvotes"],
        "downvotes"     : ANALYTICS["downvotes"],
        "top_intents"   : top,
        "model_acc"     : f"{MODEL_ACC:.1%}" if MODEL_ACC else "keyword-only",
        "model_ready"   : MODEL_READY,
        "uptime"        : f"{h:02d}h {m:02d}m {s:02d}s",
    })


@app.route("/profile")
def profile():
    return jsonify(STUDENT_PROFILE)


@app.route("/health")
def health():
    """Render health check — must return 200."""
    return jsonify({"status": "ok", "bot": "FUTObot", "version": "2.0"})


# =============================================================
# SECTION 10 — ENTRY POINT
# =============================================================
if __name__ == "__main__":
    port       = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_ENV", "production") != "production"

    log.info("=" * 55)
    log.info("  FUTO STUDENT CHATBOT v2.0")
    log.info(f"  Student : {STUDENT_PROFILE['full_name']}")
    log.info(f"  Reg No  : {STUDENT_PROFILE['reg_number']}")
    log.info(f"  Port    : {port}")
    log.info(f"  http://127.0.0.1:{port}")
    log.info("=" * 55)

    app.run(host="0.0.0.0", port=port, debug=debug_mode)