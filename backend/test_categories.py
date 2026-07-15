

# test_categories.py
import pickle
import numpy as np

# Load model
with open('complaint_classifier.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

test_complaints = [
    # Should go to ANTI-RAGGING
    ("seniors demanding money from juniors in hostel", "anti-ragging"),
    ("cyber bullying on whatsapp groups with fake profiles", "anti-ragging"),
    ("verbal abuse and name-calling by classmates", "anti-ragging"),
    
    # Should go to SECURITY
    ("fire extinguishers expired in chemistry lab", "security"),
    ("cctv cameras not working in parking area", "security"),
    ("theft of laptops from computer lab", "security"),
    
    # Should go to MAINTENANCE
    ("air conditioning broken in library", "maintenance"),
    ("water pipe leaking in hostel bathroom", "maintenance"),
    ("electrical short circuit in classroom", "maintenance"),
    
    # Should go to MEDICAL
    ("medical center doctor not available", "medical"),
    ("first aid kit empty in sports ground", "medical"),
    ("students with fever not getting treatment", "medical"),
]

print("🧪 TESTING CATEGORY DISTINCTION")
print("="*50)

for text, expected in test_complaints:
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    proba = np.max(model.predict_proba(X))
    
    if pred == expected:
        print(f"✅ '{text[:40]}...' -> {pred} ({proba:.1%})")
    else:
        print(f"❌ '{text[:40]}...' -> {pred} (expected: {expected}) ({proba:.1%})")

print("="*50)
input("Press Enter to exit...")