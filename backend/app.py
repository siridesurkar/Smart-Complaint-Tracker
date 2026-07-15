
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
import re
from datetime import datetime, timedelta, timezone
import sqlite3
import uuid  # Fixed import - removed the alias

from datetime import datetime, timedelta, timezone

# Set your timezone
IST = timezone(timedelta(hours=5, minutes=30))  # India Standard Time
# Or use: UTC = timezone.utc

app = Flask(__name__)
CORS(app)

# Load your saved model and vectorizer
with open('complaint_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def is_valid_complaint(text):
    """Check if the complaint text is meaningful"""
    clean_text = re.sub(r'\s+', ' ', text.strip()).lower()
    
    if len(clean_text) < 10:
        return False, "Complaint too short. Please provide more details."
    
    if re.search(r'(.)\1{3,}', clean_text):
        return False, "Please provide a meaningful complaint description."
    
    words = clean_text.split()
    if len(words) < 2:
        return False, "Please provide more details about your complaint."
    
    word_pattern = re.compile(r'[a-z]{3,}')
    meaningful_words = [word for word in words if word_pattern.match(word)]
    
    if len(meaningful_words) < 1:
        return False, "Please describe your issue using proper words."
    
    return True, "Valid complaint"

def detect_urgency(complaint_text):
    """
    Detect urgency level based on keywords and context
    Returns: level (high/medium/low), color (hex), icon, deadline (hours)
    """
    text_lower = complaint_text.lower()
    
    # Urgency keywords with weights
    urgency_keywords = {
        "high": {
            "words": ["fire", "emergency", "blood", "bleeding", "attack", "danger", 
                     "accident", "urgent", "immediate", "critical", "life", "safety",
                     "suicide", "heart", "unconscious", "rape", "molest", "assault",
                     "electrocution", "collapse", "poison", "choking", "drown",
                     "burning", "explosion", "gas leak", "short circuit",
                     "ambulance", "police", "help", "save", "now", "asap"],
            "weight": 10
        },
        "medium": {
            "words": ["broken", "not working", "leak", "damage", "stop", "wont start",
                     "malfunction", "issue", "problem", "stuck", "blocked", "overflow",
                     "flood", "smoke", "spark", "burn", "injury", "hurt", "pain",
                     "fever", "vomit", "food poison", "theft", "stolen", "missing",
                     "cracked", "shattered", "torn", "ripped", "damaged"],
            "weight": 5
        },
        "low": {
            "words": ["suggest", "improve", "better", "request", "could", "maybe",
                     "please", "hope", "wish", "annoying", "inconvenient", "bother",
                     "feedback", "idea", "recommend", "advice", "suggestion",
                     "nice to have", "if possible", "when convenient"],
            "weight": 1
        }
    }
    
    # Calculate urgency score
    score = 0
    
    for level, data in urgency_keywords.items():
        for word in data["words"]:
            if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
                score += data["weight"]
                if level == "high":
                    score += 5
    
    # Special case: "fire" should be very high priority
    if "fire" in text_lower:
        score += 20
    
    # Emergency indicators
    emergency_indicators = ["help", "save", "now", "quick", "fast", "asap", "stat", "emergency"]
    for indicator in emergency_indicators:
        if indicator in text_lower:
            score += 15
    
    # Check for ALL CAPS
    if complaint_text.isupper() or sum(1 for c in complaint_text if c.isupper()) > len(complaint_text) * 0.3:
        score += 10
    
    # Check for multiple exclamation marks
    exclamation_count = complaint_text.count('!')
    if exclamation_count >= 1:
        score += exclamation_count * 3
    
    # Check for question marks in emergency context
    if "?" in complaint_text and any(word in text_lower for word in ["help", "what", "how", "emergency"]):
        score += 5
    
    # Check for time-sensitive words
    time_words = ["now", "immediate", "today", "right now", "asap", "urgent"]
    for word in time_words:
        if word in text_lower:
            score += 8
    
    # Check for danger/risk words
    danger_words = ["danger", "risk", "unsafe", "hazard", "threat", "emergency"]
    for word in danger_words:
        if word in text_lower:
            score += 12
    
    # Determine urgency level
    if score >= 15:
        return {
            "level": "high",
            "color": "#dc3545",
            "icon": "🚨",
            "deadline_hours": 1,
            "message": "EMERGENCY - Immediate action required!",
            "priority": 1,
            "score": score,
            "detected_reasons": get_urgency_reasons(text_lower, score)
        }
    elif score >= 8:
        return {
            "level": "medium",
            "color": "#fd7e14",
            "icon": "⚠️",
            "deadline_hours": 12,
            "message": "Priority - Resolve within 12 hours",
            "priority": 2,
            "score": score,
            "detected_reasons": get_urgency_reasons(text_lower, score)
        }
    else:
        return {
            "level": "low",
            "color": "#28a745",
            "icon": "📝",
            "deadline_hours": 72,
            "message": "Normal - Resolve within 3 days",
            "priority": 3,
            "score": score,
            "detected_reasons": get_urgency_reasons(text_lower, score)
        }

def get_urgency_reasons(text, score):
    """Explain why this urgency level was assigned"""
    reasons = []
    
    if "fire" in text:
        reasons.append("🔥 'Fire' detected - Highest priority")
    
    emergency_words = ["emergency", "urgent", "immediate", "asap"]
    for word in emergency_words:
        if word in text:
            reasons.append(f"🚨 '{word}' indicates emergency")
    
    if text.isupper() or sum(1 for c in text if c.isupper()) > len(text) * 0.3:
        reasons.append("🔊 ALL CAPS indicates urgency")
    
    exclamation_count = text.count('!')
    if exclamation_count > 0:
        reasons.append(f"❗ {exclamation_count} exclamation marks")
    
    danger_words = ["danger", "risk", "unsafe", "hazard"]
    for word in danger_words:
        if word in text:
            reasons.append(f"⚠️ Safety concern: '{word}'")
    
    return reasons[:3]

def get_confidence_level(confidence):
    """Determine confidence level"""
    if confidence > 0.7:
        return "high", "✅ Confident classification"
    elif confidence > 0.5:
        return "medium", "⚠️  Moderately confident - review suggested"
    elif confidence > 0.3:
        return "low", "❓ Low confidence - please provide more specific details"
    else:
        return "very_low", "🤔 Unable to categorize - please rephrase"

# ===== DATABASE SETUP =====
# ===== DATABASE SETUP =====
def init_db():
    """Create database and tables if they don't exist"""
    conn = sqlite3.connect('complaints.db')
    cursor = conn.cursor()
    
    # Create complaints table - UPDATED with proper timezone handling
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS complaints (
        id TEXT PRIMARY KEY,
        complaint_text TEXT NOT NULL,
        category TEXT NOT NULL,
        urgency_level TEXT NOT NULL,
        status TEXT DEFAULT 'submitted',
        submitted_at TIMESTAMP DEFAULT (datetime('now', 'localtime')), 
        updated_at TIMESTAMP DEFAULT (datetime('now', 'localtime')),  
        deadline TIMESTAMP,
        assigned_to TEXT DEFAULT 'Not Assigned',
        resolution_notes TEXT DEFAULT ''
    )
    ''')
    
    # Create status_updates table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS status_updates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        complaint_id TEXT,
        status TEXT,
        updated_by TEXT DEFAULT 'system',
        notes TEXT,
        created_at TIMESTAMP DEFAULT (datetime('now', 'localtime')),  
        FOREIGN KEY (complaint_id) REFERENCES complaints (id)
    )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Database initialized")

# Initialize database
init_db()
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/track')
def track():
    return render_template('track.html')
@app.route('/submit-complaint', methods=['POST'])
def submit_complaint():
    data = request.json
    complaint_text = data.get('complaint_text', '').strip()

    if not complaint_text:
        return jsonify({
            'error': 'No complaint text received',
            'suggestion': 'Please describe your issue in the complaint box'
        }), 400

    # Validate complaint quality
    is_valid, message = is_valid_complaint(complaint_text)
    if not is_valid:
        return jsonify({
            'error': 'Invalid complaint format',
            'message': message
        }), 400

    try:
        # Detect urgency
        urgency = detect_urgency(complaint_text)
        
        # Categorize
        X_input = vectorizer.transform([complaint_text])
        category = model.predict(X_input)[0]
        confidence = np.max(model.predict_proba(X_input))
        
        # Get confidence level
        conf_level, conf_message = get_confidence_level(confidence)
        
        # Calculate deadline with local timezone - FIXED HERE
        from datetime import datetime, timedelta
        import pytz
        
        # Get current time in IST (India Standard Time)
        ist = pytz.timezone('Asia/Kolkata')
        current_time = datetime.now(ist)
        
        # Calculate deadline
        deadline = current_time + timedelta(hours=urgency["deadline_hours"])
        
        # Generate unique complaint ID
        complaint_id = str(uuid.uuid4())[:8].upper()
        
        # Format timestamps for display
        submitted_at_str = current_time.strftime("%d %b %Y, %I:%M %p")
        deadline_str = deadline.strftime("%d %b %Y, %I:%M %p")
        
        # Prepare response
        response = {
            'complaint_id': complaint_id,
            'category': category,
            'confidence': float(confidence),
            'confidence_level': conf_level,
            'urgency': urgency,
            'submitted_at': submitted_at_str,  # CHANGED: Formatted string
            'deadline': deadline_str,          # CHANGED: Formatted string
            'message': 'Complaint submitted successfully'
        }
        
        # Add urgency-specific recommendations
        if urgency["level"] == "high":
            response['action_required'] = "🚨 EMERGENCY PROTOCOL ACTIVATED"
            response['instructions'] = [
                "Security/Medical team has been notified",
                "Please stay at safe location if this is a physical emergency",
                "Emergency contact: 108 (Ambulance) / 101 (Fire) / 100 (Police)"
            ]
        elif urgency["level"] == "medium":
            response['action_required'] = "Priority complaint - Escalated to department head"
            response['instructions'] = [
                f"Assigned to {category} department",
                f"Expected resolution: Within {urgency['deadline_hours']} hours",
                "You will receive updates via email"
            ]
        else:
            response['action_required'] = "Added to regular queue"
            response['instructions'] = [
                f"Expected resolution: Within {urgency['deadline_hours']} hours",
                "You can check status using your Complaint ID",
                "Thank you for your patience"
            ]
        
        # Save to database - UPDATED with local time
        try:
            conn = sqlite3.connect('complaints.db')
            cursor = conn.cursor()
            
            # Store timestamps in ISO format for consistency
            cursor.execute('''
                INSERT INTO complaints 
                (id, complaint_text, category, urgency_level, status, submitted_at, deadline, assigned_to)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                complaint_id,
                complaint_text,
                category,
                urgency["level"],
                'submitted',
                current_time.strftime("%Y-%m-%d %H:%M:%S"),  # ISO format for DB
                deadline.strftime("%Y-%m-%d %H:%M:%S"),      # ISO format for DB
                category + " Department"
            ))
            
            # Log initial status
            cursor.execute('''
                INSERT INTO status_updates (complaint_id, status, updated_by, notes)
                VALUES (?, ?, ?, ?)
            ''', (complaint_id, 'submitted', 'system', 'Complaint submitted via portal'))
            
            conn.commit()
            conn.close()
            
        except Exception as db_error:
            print(f"⚠️ Database error: {db_error}")
            # Don't fail the request, just log the error
        
        return jsonify(response)

    except Exception as e:
        print(f"Error in submit_complaint: {e}")
        return jsonify({
            'error': 'Error processing your complaint',
            'message': str(e)
        }), 500
@app.route('/update-status', methods=['POST'])
def update_status():
    """Admin endpoint to update complaint status"""
    try:
        data = request.json
        complaint_id = data.get('complaint_id')
        new_status = data.get('status')
        admin_name = data.get('admin_name', 'Admin')
        notes = data.get('notes', '')
        
        if not complaint_id or not new_status:
            return jsonify({'error': 'Missing complaint_id or status'}), 400
        
        # Valid statuses
        valid_statuses = ['submitted', 'under_review', 'in_progress', 'resolved']
        if new_status not in valid_statuses:
            return jsonify({'error': f'Invalid status. Must be one of: {valid_statuses}'}), 400
        
        conn = sqlite3.connect('complaints.db')
        cursor = conn.cursor()
        
        # Update complaint status
        cursor.execute('''
            UPDATE complaints 
            SET status = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (new_status, complaint_id))
        
        # Log the status change
        cursor.execute('''
            INSERT INTO status_updates (complaint_id, status, updated_by, notes)
            VALUES (?, ?, ?, ?)
        ''', (complaint_id, new_status, admin_name, notes))
        
        conn.commit()
        
        # Get updated complaint info
        cursor.execute('''
            SELECT * FROM complaints WHERE id = ?
        ''', (complaint_id,))
        
        complaint = cursor.fetchone()
        conn.close()
        
        if complaint:
            return jsonify({
                'success': True,
                'message': f'Status updated to {new_status}',
                'complaint_id': complaint_id,
                'new_status': new_status,
                'updated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        else:
            return jsonify({'error': 'Complaint not found'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/check-status/<path:complaint_id>', methods=['GET'])
def check_status(complaint_id):
    """Check the current status of a complaint"""
    try:
        # Decode the complaint_id if it's URL encoded
        import urllib.parse
        complaint_id = urllib.parse.unquote(complaint_id)
        
        conn = sqlite3.connect('complaints.db')
        cursor = conn.cursor()
        
        # Get complaint details
        cursor.execute('''
            SELECT id, complaint_text, category, urgency_level, status, 
                   submitted_at, deadline, assigned_to
            FROM complaints 
            WHERE id = ?
        ''', (complaint_id,))
        
        complaint = cursor.fetchone()
        
        if not complaint:
            return jsonify({'error': 'Complaint not found'}), 404
        
        # Get status history
        cursor.execute('''
            SELECT status, updated_by, notes, created_at
            FROM status_updates
            WHERE complaint_id = ?
            ORDER BY created_at DESC
        ''', (complaint_id,))
        
        history = cursor.fetchall()
        conn.close()
        
        # Helper function to format date properly
        def format_datetime(dt_str):
            if not dt_str:
                return "Not available"
            try:
                # Parse the datetime string
                dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                # Format for Indian timezone display
                return dt.strftime("%d %b %Y, %I:%M %p")
            except:
                return dt_str  # Return as-is if parsing fails
        
        # Format response with proper timestamps
        response = {
            'complaint_id': complaint[0],
            'category': complaint[2],
            'urgency': complaint[3],
            'current_status': complaint[4],
            'submitted_at': format_datetime(complaint[5]),  # FORMATTED
            'deadline': format_datetime(complaint[6]),      # FORMATTED
            'assigned_to': complaint[7],
            'status_history': [
                {
                    'status': h[0],
                    'updated_by': h[1],
                    'notes': h[2],
                    'timestamp': format_datetime(h[3])      # FORMATTED
                }
                for h in history
            ]
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500   
    
@app.route('/analyze-urgency', methods=['POST'])
def analyze_urgency():
    data = request.json
    complaint_text = data.get('complaint_text', '').strip()
    
    if not complaint_text:
        return jsonify({'error': 'No text provided'}), 400
    
    urgency = detect_urgency(complaint_text)
    
    # Get real-time suggestions based on urgency
    suggestions = {
        "high": [
            "⚠️ This appears to be an EMERGENCY",
            "Call campus security immediately if needed: 911",
            "Move to a safe location if in danger",
            "Help is on the way - hold tight"
        ],
        "medium": [
            "This is a priority issue",
            "Will be addressed within 24 hours",
            "Consider calling department directly for faster response"
        ],
        "low": [
            "Thank you for your feedback",
            "This will be processed in regular queue",
            "Expected resolution: 3 working days"
        ]
    }
    
    return jsonify({
        'urgency': urgency,
        'suggestions': suggestions[urgency['level']],
        'immediate_actions': get_immediate_actions(urgency['level'])
    })

def get_immediate_actions(urgency_level):
    """Get immediate actions based on urgency"""
    actions = {
        "high": [
            "Notify campus security",
            "Alert medical emergency team",
            "Send SMS to department heads",
            "Activate emergency protocol"
        ],
        "medium": [
            "Assign to department queue",
            "Notify department head",
            "Set 24-hour reminder"
        ],
        "low": [
            "Add to general queue",
            "Schedule for next working day",
            "Send confirmation email"
        ]
    }
    return actions.get(urgency_level, [])

@app.route('/get-all-complaints', methods=['GET'])
def get_all_complaints():
    """Get all complaints for admin panel"""
    try:
        conn = sqlite3.connect('complaints.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, complaint_text, category, urgency_level, status, 
                   submitted_at, deadline, assigned_to
            FROM complaints
            ORDER BY submitted_at DESC
        ''')
        
        complaints = cursor.fetchall()
        conn.close()
        
        # Convert to list of dictionaries
        complaint_list = []
        for comp in complaints:
            complaint_list.append({
                'id': comp[0],
                'complaint_text': comp[1],
                'category': comp[2],
                'urgency_level': comp[3],
                'status': comp[4],
                'submitted_at': comp[5],
                'deadline': comp[6],
                'assigned_to': comp[7]
            })
        
        return jsonify(complaint_list)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/admin')
def admin_panel():
    return open('admin.html').read()

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')