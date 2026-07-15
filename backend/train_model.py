

# train_model.py - WORKING VERSION WITH 40 EXAMPLES PER CATEGORY
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

# ========== 40 EXAMPLES PER CATEGORY ==========

complaints = [
    # 1. MAINTENANCE (40)
    "fan not working in classroom", "ceiling fan broken", "fan stopped working",
    "broken window in hall", "window glass shattered", "window wont close",
    "water pipe leaking", "pipe burst in bathroom", "toilet flush not working",
    "flush system broken", "air conditioner not cooling", "ac not working",
    "lights flickering", "light bulbs need replacement", "classroom furniture broken",
    "chair legs broken", "desk damaged", "power outlet not working",
    "electrical socket dead", "plug point not functioning", "water cooler broken",
    "water dispenser malfunction", "elevator stuck", "lift not moving",
    "elevator door stuck", "peeling paint on walls", "wall paint coming off",
    "door lock broken", "door handle broken", "roof leaking",
    "floor tiles cracked", "blackboard broken", "projector not working",
    "speaker system not working", "microphone not working", "clock not working",
    "switchboard damaged", "fan regulator broken", "exhaust fan not working",
    "ventilation not working",
    
    # 2. HOSTEL (40)
    "no hot water in hostel", "hostel geyser not working", "cold water in hostel",
    "dirty hostel room", "hostel room not cleaned", "hostel bathroom dirty",
    "bed bugs in hostel", "hostel mattress dirty", "insects in hostel room",
    "hostel wifi not working", "hostel internet down", "hostel network issues",
    "hostel laundry machine broken", "washing machine not working",
    "poor ventilation in hostel", "hostel room stuffy", "hostel security issues",
    "unauthorized entry in hostel", "hostel safety concerns", "hostel food bad",
    "mess food not tasty", "hostel meals unhygienic", "roommate conflict",
    "hostel roommate issues", "roommate harassment", "hostel drainage blocked",
    "clogged sink in hostel", "hostel toilet clogged", "hostel shower not working",
    "hostel mirror broken", "hostel cupboard broken", "hostel lights not working",
    "hostel fan not working", "hostel ac not working", "hostel bed broken",
    "hostel chair broken", "hostel table broken", "hostel window broken",
    "hostel door lock broken", "hostel common room dirty",
    
    # 3. SANITATION (40)
    "unhygienic toilets", "dirty restrooms", "toilets not cleaned",
    "garbage not collected", "trash accumulation", "waste disposal issues",
    "foul smell from bathrooms", "bad odor in toilets", "stench from toilets",
    "urinals blocked", "toilet seats broken", "restroom doors broken",
    "no soap in bathrooms", "handwash not available", "sanitizer empty",
    "dirty cafeteria tables", "unclean dining area", "food spillage not cleaned",
    "dirty corridors", "dirty classrooms", "dusty furniture",
    "mold in bathrooms", "fungus on walls", "dirty water tanks",
    "dirty windows", "dirty floors", "spider webs everywhere",
    "mosquito breeding", "rats in campus", "cockroaches in kitchen",
    "dirty drinking water", "contaminated water", "dirty utensils",
    "unclean kitchen", "dirty food counters", "dirty wash basins",
    "overflowing dustbins", "garbage smell", "dirty parking area",
    "unclean gym",
    
    # 4. ANTI-RAGGING (40)
    "ragging by seniors", "seniors forcing juniors", "ragging in hostel",
    "eve teasing", "girls being harassed", "verbal harassment",
    "seniors bullying juniors", "physical intimidation", "cyber bullying",
    "online harassment", "physical assault threat", "threatened by students",
    "mental harassment", "psychological bullying", "emotional abuse",
    "discrimination based on state", "regional bias", "language discrimination",
    "social media harassment", "fake profile created", "online defamation",
    "peer pressure", "forced participation", "coerced into wrong deeds",
    "isolation of students", "group boycott", "social exclusion",
    "verbal abuse", "foul language used", "threatening words",
    "extortion from juniors", "money demanded", "assignments forced",
    "notes forced to give", "laboratory work forced", "practical files forced",
    "dress code bullying", "appearance teasing", "weight shaming",
    "height mocking",
    
    # 5. ADMINISTRATION (40)
    "lost id card", "student id missing", "identity card lost",
    "long queue at admin", "waiting time too long", "admin counter crowded",
    "document verification delay", "certificate attestation slow", "paperwork pending",
    "fee receipt not issued", "payment confirmation missing", "fee slip not received",
    "transfer certificate delay", "tc processing slow", "attendance record wrong",
    "marks not updated", "record mismatch", "bonafide certificate urgent",
    "college certificate needed", "admin staff rude", "office personnel disrespectful",
    "clerk misbehaving", "application form not available", "scholarship delay",
    "fee refund delay", "library card issue", "hostel allotment delay",
    "bus pass not issued", "parking sticker missing", "no objection certificate delay",
    "character certificate delay", "migration certificate delay", "duplicate marksheet",
    "degree certificate delay", "convocation certificate", "alumni registration",
    "training certificate", "internship letter", "project approval delay",
    "thesis submission issue",
    
    # 6. ACADEMIC (40)
    "assignment deadline extension", "need more time for project", "submission date extend",
    "exam timetable confusion", "schedule not clear", "exam dates conflicting",
    "course material not available", "study notes missing", "textbooks not provided",
    "professor not available", "teacher absent", "faculty unavailable",
    "difficulty understanding subject", "concepts not clear", "need extra classes",
    "project submission extension", "practical file deadline", "lab work submission",
    "seminar postponement", "viva voce reschedule", "internal marks dispute",
    "assignment marks wrong", "project evaluation unfair", "lab records misplaced",
    "tutorial classes needed", "doubt clearing session", "makeup class required",
    "guest lecture request", "workshop organization", "industrial visit planning",
    "research guidance needed", "thesis supervisor change", "journal paper submission",
    "conference participation", "project funding issue", "lab equipment training",
    "software training needed", "coding practice sessions", "presentation skills",
    "communication skills class",
    
    # 7. MESS (40)
    "food served cold", "meals not hot", "cold food in mess",
    "contaminated food", "food poisoning case", "stale food served",
    "poor hygiene in kitchen", "unclean cooking area", "kitchen staff not clean",
    "limited vegetarian options", "no veg food choice", "vegetarian menu limited",
    "overpriced food", "food too expensive", "unreasonable pricing",
    "insufficient food quantity", "less food served", "not enough portions",
    "repeated menu", "same food daily", "menu variety needed",
    "food taste bad", "too much oil", "too spicy food",
    "bland food", "undercooked food", "overcooked food",
    "non-veg mixing in veg", "egg in vegetarian food", "fish smell in veg area",
    "dirty plates", "unclean utensils", "rusty spoons",
    "broken plates", "cracked glasses", "dirty water glasses",
    "no drinking water", "water not purified", "unhygienic dining",
    "flies in mess",
    
    # 8. LIBRARY (40)
    "required books not available", "reference book missing", "textbook not in library",
    "library closing early", "reading room timings reduced", "library hours shortened",
    "noise in library", "disturbance in reading area", "students talking loudly",
    "insufficient copies", "book damaged", "pages torn",
    "journal not available", "research paper missing", "thesis not found",
    "internet not working in library", "library computers slow", "scanner not working",
    "photocopier broken", "printer not working", "charging points not working",
    "insufficient seating", "no study tables", "chairs uncomfortable",
    "air conditioning not working", "lights not working", "fan not working",
    "water cooler empty", "washroom dirty", "library staff rude",
    "book issue delay", "return policy strict", "fine amount high",
    "membership issue", "library card problem", "digital library access",
    "e-journals not accessible", "database subscription expired", "online resources down",
    "study room booking issue",
    
    # 9. TRANSPORT (40)
    "college bus late", "bus not on time", "transport delay",
    "bus overcrowded", "too many students in bus", "bus packed full",
    "bus broke down", "bus accident", "bus damaged",
    "bus driver rash driving", "driver drunk", "driver misbehaving",
    "conductor rude", "conductor charging extra", "ticket issue",
    "bus pass problem", "route change issue", "bus stop missing",
    "no bus for route", "insufficient buses", "bus frequency low",
    "parking space full", "no parking space", "vehicle parking issue",
    "bicycle stolen", "scooter stolen", "car damaged in parking",
    "parking fee issue", "parking permit", "visitor parking",
    "two-wheeler parking", "four-wheeler parking", "handicap parking",
    "parking security", "cctv not working in parking", "parking lights broken",
    "parking gate broken", "parking area dirty", "parking slot allocation",
    "parking charge dispute",
    
    # 10. IT SUPPORT (40)
    "wifi not working", "internet connection down", "network issues",
    "computer systems slow", "pc hanging", "systems not responding",
    "software not installed", "antivirus expired", "operating system issue",
    "projector not connecting", "smart board not working", "audio system problem",
    "website not loading", "college portal down", "online exam portal issue",
    "email not working", "student portal login issue", "password reset problem",
    "database error", "server down", "website maintenance",
    "online class issue", "zoom not working", "google meet problem",
    "recording not available", "lecture videos missing", "online submission portal",
    "coding software issue", "matlab not working", "autocad problem",
    "programming lab computers", "simulation software", "virtual lab access",
    "hacking attempt", "cyber security", "data backup issue",
    "printer network issue", "scanner software", "biometric system",
    "attendance software",
    
    # 11. MEDICAL (40)
    "medical center closed", "clinic not open", "doctor not available",
    "no first aid", "emergency kit empty", "medical supplies missing",
    "ambulance not available", "ambulance driver absent", "emergency contact",
    "medicine stock finished", "basic medicines not available", "bandage not available",
    "thermometer broken", "bp machine not working", "stethoscope missing",
    "injection not available", "drip set missing", "oxygen cylinder empty",
    "wheelchair broken", "stretcher missing", "first aid training needed",
    "health checkup camp", "vaccination camp", "blood donation camp",
    "psychologist needed", "counseling service", "mental health support",
    "dental checkup", "eye checkup", "general health camp",
    "fever cases", "vomiting cases", "food poisoning cases",
    "accident cases", "fracture cases", "burn cases",
    "allergy cases", "asthma attack", "heart problem",
    "diabetes emergency",
    
    # 12. SPORTS (40)
    "sports injury", "got hurt in basketball", "football injury",
    "broken sports equipment", "cricket bat cracked", "hockey stick broken",
    "badminton racket string broken", "tennis balls finished", "volleyball net torn",
    "chess board missing", "carrom board broken", "table tennis paddle broken",
    "gym equipment broken", "treadmill not working", "exercise machine damaged",
    "swimming pool dirty", "pool water not clean", "pool maintenance",
    "sports ground uneven", "cricket pitch damaged", "football ground maintenance",
    "basketball court net torn", "volleyball court net broken", "tennis court crack",
    "athletics track repair", "jumping pit maintenance", "throwing area safety",
    "sports kit not issued", "jersey not given", "shoes not provided",
    "coach not available", "trainer absent", "sports teacher missing",
    "tournament organization", "inter-college match", "sports fest",
    "practice timing conflict", "team selection issue", "captain selection",
    "sports scholarship",
    
    # 13. SECURITY (40)
    "security guard rude", "guard misbehaving", "security personnel abusive",
    "campus gate locked early", "main gate closed before time", "entry denied",
    "theft in campus", "mobile stolen", "laptop stolen",
    "wallet stolen", "bag stolen", "books stolen",
    "unauthorized entry", "strangers in campus", "suspicious persons",
    "eve teasing cases", "girls security", "women safety",
    "cctv not working", "security cameras broken", "surveillance issue",
    "fire safety equipment", "fire extinguisher expired", "fire alarm not working",
    "emergency exit blocked", "safety drills needed", "disaster management",
    "night security", "hostel security", "library security",
    "lab security", "computer lab safety", "equipment theft",
    "vehicle theft", "bicycle theft", "scooter theft",
    "parking security", "classroom security", "exam hall security",
    "fest security",
    
    # 14. DISCIPLINE (40)
    "noise in classroom", "students disturbing lecture", "classroom disruption",
    "smoking in campus", "cigarette smoking prohibited area", "alcohol consumption",
    "drug abuse", "substance abuse", "rave party",
    "late night parties", "loud music", "disturbing neighbors",
    "dress code violation", "uniform not worn", "inappropriate dressing",
    "mobile phone misuse", "phone in exam hall", "cheating cases",
    "plagiarism cases", "assignment copying", "project copying",
    "fake medical certificates", "fake leave letters", "attendance manipulation",
    "false complaints", "fake identity", "impersonation",
    "vandalism", "property damage", "wall graffiti",
    "tree cutting", "garden destruction", "bench breaking",
    "waste disposal violation", "littering", "spitting",
    "public display of affection", "indecent behavior", "moral policing",
    "cultural misconduct",
    
    # 15. EXAM CELL (40)
    "hall ticket not issued", "admit card missing", "exam form issue",
    "exam date change", "schedule conflict", "clashing exam timings",
    "exam center change", "venue problem", "seating arrangement",
    "question paper leak", "paper pattern change", "syllabus not covered",
    "evaluation mistake", "marksheet error", "grade calculation wrong",
    "revaluation process", "answer script copy", "photocopy of answer sheet",
    "supplementary exam date", "backlog exam schedule", "arrears clearance",
    "project viva date", "practical exam schedule", "internal assessment marks",
    "external examiner issue", "invigilator problem", "malpractice case",
    "exam postponement", "cancellation of exam", "online exam technical issue",
    "offline exam center", "distance exam problem", "semester exam fee",
    "exam fee refund", "scholarship exam", "entrance exam coaching",
    "competitive exam preparation", "gate exam support", "cat exam facility",
    "placement test",
]

categories = []
# Add 40 for each of 15 categories
for category in ["maintenance", "hostel", "sanitation", "anti-ragging", "administration", 
                 "academic", "mess", "library", "transport", "it support", 
                 "medical", "sports", "security", "discipline", "exam cell"]:
    categories.extend([category] * 40)

# ========== VERIFICATION ==========
print("="*70)
print("🤖 SMART COMPLAINT CLASSIFIER - 600 EXAMPLES")
print("="*70)
print(f"Total Complaints: {len(complaints)}")
print(f"Total Categories: {len(categories)}")
print(f"Categories: {set(categories)}")

if len(complaints) != len(categories):
    print(f"❌ ERROR: Mismatch! Fixing...")
    min_len = min(len(complaints), len(categories))
    complaints = complaints[:min_len]
    categories = categories[:min_len]
    print(f"✅ Fixed to {min_len} items each")
else:
    print("✅ Perfect Match!")

# ========== TRAIN MODEL ==========
print("\n🔄 Training Model...")

vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
X = vectorizer.fit_transform(complaints)

model = LogisticRegression(max_iter=2000, random_state=42)
model.fit(X, categories)

# ========== SAVE MODEL ==========
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
    
with open("complaint_classifier.pkl", "wb") as f:
    pickle.dump(model, f)

print("💾 Model saved successfully!")

# ========== TEST ==========
print("\n" + "="*70)
print("🧪 TESTING ALL CATEGORIES")
print("="*70)

test_cases = [
    ("fan not working", "maintenance"),
    ("hostel room dirty", "hostel"),
    ("toilets unhygienic", "sanitation"),
    ("ragging by seniors", "anti-ragging"),
    ("id card lost", "administration"),
    ("assignment deadline", "academic"),
    ("food cold in mess", "mess"),
    ("library books missing", "library"),
    ("bus late arrival", "transport"),
    ("wifi not working", "it support"),
    ("medical center closed", "medical"),
    ("sports equipment broken", "sports"),
    ("security guard rude", "security"),
    ("noise in classroom", "discipline"),
    ("exam hall ticket", "exam cell"),
]

correct = 0
for text, expected in test_cases:
    X_test = vectorizer.transform([text])
    pred = model.predict(X_test)[0]
    proba = np.max(model.predict_proba(X_test)) * 100
    
    if pred == expected:
        correct += 1
        print(f"✅ '{text}' -> {pred} ({proba:.1f}%)")
    else:
        print(f"❌ '{text}' -> {pred} (expected: {expected}) ({proba:.1f}%)")

accuracy = (correct / len(test_cases)) * 100
print("="*70)
print(f"🎯 Accuracy: {accuracy:.1f}% ({correct}/{len(test_cases)} correct)")

if accuracy == 100:
    print("🚀 PERFECT! Model is ready to use!")
else:
    print("⚠️ Some errors found. Check training data.")

print("\n✅ To use: Run 'python app.py' and test with complaints!")
print("="*70)