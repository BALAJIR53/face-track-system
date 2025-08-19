import os
import mysql.connector
import cv2
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, date
from flask import Flask, request, render_template, redirect, url_for, session, send_file, flash, jsonify
from sklearn.neighbors import KNeighborsClassifier
from twilio.rest import Client
import pyttsx3  # For voice announcements
import random
import glob
import json
import csv

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Twilio credentials
account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
phone_number = os.getenv("TWILIO_PHONE_NUMBER")

# Constants
nimgs = 10
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")
csv_filename = f'Attendance/Attendance-{datetoday}.csv'

# Create necessary directories
os.makedirs('Attendance', exist_ok=True)
os.makedirs('static/faces', exist_ok=True)

# Ensure today's attendance file exists
if not os.path.isfile(csv_filename):
    with open(csv_filename, 'w') as f:
        f.write('Name,Roll,Time,LastAttendanceTime,Status,Department\n')

# Global variable for pending approvals
pending_attendance = []

# Load Haar Cascade for face detection
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Helper Functions

def totalreg():
    """Count total registered users."""
    return len(os.listdir('static/faces'))

def extract_faces(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return faces

def identify_face(face, model):
    try:
        prediction = model.predict(face)
        confidence = model.predict_proba(face)
        if max(confidence[0]) < 0.6:  # Set a confidence threshold
            return 'unknown'
        return prediction[0]
    except Exception as e:
        return 'unknown'

def train_model():
    faces = []
    labels = []
    
    user_folders = os.listdir('static/faces')
    if not user_folders:
        print("[ERROR] No user folders found in 'static/faces'.")
        return

    for label, folder in enumerate(user_folders):
        folder_path = os.path.join('static/faces', folder)
        if not os.path.isdir(folder_path):
            print(f"[WARNING] Skipping non-directory: {folder_path}")
            continue
        
        face_images = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]
        if not face_images:
            print(f"[INFO] No face images found in: {folder_path}, skipping.")
            continue
        
        for filename in face_images:
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[WARNING] Skipping invalid image: {img_path}")
                continue
            
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face)
            labels.append(label)

    if not faces:
        print("[ERROR] No valid face images found. Training aborted.")
        return

    # Convert faces to numpy array of correct shape (num_samples, height, width)
    faces_np = np.array(faces)
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces_np, np.array(labels))
    recognizer.save("model.yml")
    
    print(f"[INFO] Model trained successfully with {len(set(labels))} users.")

def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return faces




def extract_attendance():
    """Extract attendance data from the latest CSV file."""
    try:
        csv_filename = get_latest_attendance_file('attendance_records')
        if not csv_filename or not os.path.exists(csv_filename):
            print("CSV file not found.")
            return [], [], [], [], [], []

        df = pd.read_csv(csv_filename)
        print("Loaded CSV:", csv_filename)

        names = df['Name'].tolist() if 'Name' in df.columns else []
        rolls = df['Roll'].tolist() if 'Roll' in df.columns else []
        times = df['Time'].tolist() if 'Time' in df.columns else []
        last_times = df['LastAttendanceTime'].tolist() if 'LastAttendanceTime' in df.columns else []
        statuses = df['Status'].tolist() if 'Status' in df.columns else []
        departments = df['Department'].tolist() if 'Department' in df.columns else []

        return names, rolls, times, last_times, statuses, departments
    except Exception as e:
        print(f"Error reading attendance file: {e}")
        return [], [], [], [], [], []
    
def add_attendance(name, phone):
    """Add attendance to pending approvals and mark 'Present'."""
    global pending_attendance
    if isinstance(name, np.ndarray):
        name = name[0]  # Ensure it's a string

    try:
        username, userid = name.split('_')
    except ValueError:
        print(f"Invalid name format: {name}")
        return

    current_time = datetime.now()

    # Prevent duplicate attendance within an hour
    for record in pending_attendance:
        if record['Name'] == username and record['Roll'] == userid:
            last_time = datetime.strptime(record['LastAttendanceTime'], "%Y-%m-%d %H:%M:%S")
            if (current_time - last_time).seconds < 3600:  # 1 hour = 3600 seconds
                print(f"Attendance already taken for {username} within the last hour.")
                return

    record = {
        'Name': username,
        'Roll': userid,
        'Time': current_time.strftime("%H:%M:%S"),
        'LastAttendanceTime': current_time.strftime("%Y-%m-%d %H:%M:%S"),
        'Status': 'Present'
    }
    pending_attendance.append(record)
    send_sms_notification(phone, username, current_time)

def send_sms_notification(phone, username, current_time):
    """Send SMS notification."""
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = f'Hello {username}, your attendance has been recorded at {current_time.strftime("%H:%M:%S")}.'
        client.messages.create(body=message, from_=TWILIO_PHONE_NUMBER, to=+916382845186)
    except Exception as e:
        print(f"Failed to send SMS: {e}")

def announce_attendance(username):
    """Announce attendance using text-to-speech."""
    engine = pyttsx3.init()
    if username == "Unknown person detected. Please register.":
        engine.say("Unknown person detected. Please add your details to my database.")
    else:
        engine.say(f"Attendance taken for {username}.")
    engine.runAndWait()
    
   

def mark_absent_users():
    """Mark users as absent who did not take attendance."""
    names, rolls, times, last_times, statuses = extract_attendance()
    
    # Update CSV file with absent status for missing attendance
    df = pd.read_csv(csv_filename)
    for idx in range(len(df)):
        if pd.isna(df.at[idx, 'Status']) or df.at[idx, 'Status'] == "":
            df.at[idx, 'Status'] = "Absent"
    df.to_csv(csv_filename, index=False)

# MySQL Database Connection
def get_db_connection():
    return mysql.connector.connect(
        host='localhost',
        user='root',  # Your MySQL username
        password='',  # Your MySQL password
        database='userDB'  # Your database name
    )

# Function to save attendance to MySQL Database
def save_attendance_to_db(name, roll, status, department):
    conn = get_db_connection()
    cursor = conn.cursor()
    query = """
    INSERT INTO attendance (name, roll, time, last_attendance_time, status, department)
    VALUES (%s, %s, NOW(), NOW(), %s, %s)
    """
    cursor.execute(query, (name, roll, status, department))
    conn.commit()
    cursor.close()
    conn.close()
    


def get_latest_attendance_file(directory='attendance_records'):
    if not os.path.exists(directory):
        print(f"Directory '{directory}' not found.")
        return None

    files = [f for f in os.listdir(directory) if f.startswith('Attendance') and f.endswith('.csv')]
    if not files:
        return None

    # Assuming filename format: Attendance-MM_DD_YY.csv
    def file_date_key(filename):
        try:
            date_part = filename.replace('Attendance-', '').replace('.csv', '')  # "12_24_24"
            parts = date_part.split('_')
            return (int(parts[2]), int(parts[0]), int(parts[1]))  # YY, MM, DD
        except Exception:
            return (0, 0, 0)

    files.sort(key=file_date_key, reverse=True)
    return os.path.join(directory, files[0])

# Routes



@app.route('/')
def home_page():
    names, rolls, times, last_times, statuses,departments = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times,
                           last_times=last_times, statuses=statuses,departments=departments)
@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'admin123':  # Replace with your admin credentials
            session['admin_logged_in'] = True
            return redirect(url_for('admin_approval'))
        else:
            return render_template('admin_login.html', message="Invalid credentials.")
    return render_template('admin_login.html')

@app.route('/admin_approval', methods=['GET', 'POST'])
def admin_approval():
    global pending_attendance
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))

    if request.method == 'POST':
        action = request.form.get('action')
        selected_rolls = request.form.getlist('attendance[]')
        valid_rolls = [roll for roll in selected_rolls if roll]

        if action == 'approve' and valid_rolls:
            try:
                with open(csv_filename, 'a', newline='') as f:
                    for roll in valid_rolls:
                        record = next((r for r in pending_attendance if r['Roll'] == roll), None)
                        if record:
                            updated_department = request.form.get(f'department_{roll}', '').strip()
                            record['Department'] = updated_department if updated_department else "Unknown"

                            # Write in the correct order
                            f.write(f"{record['Name']},{record['Roll']},{record['Time']},{record['LastAttendanceTime']},{record['Status']},{record['Department']}\n")

                    # Remove approved records from pending
                    pending_attendance = [r for r in pending_attendance if r['Roll'] not in valid_rolls]

                flash("Attendance approved successfully.", "success")
            except Exception as e:
                flash(f"Error approving attendance: {str(e)}", "danger")

        elif action == 'reject':
            pending_attendance = [r for r in pending_attendance if r['Roll'] not in valid_rolls]
            flash("Attendance rejected successfully.", "warning")

    return render_template('admin_approval.html', pending_attendance=pending_attendance, pending_count=len(pending_attendance))

        

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/dashboard')
def dashboard():
    names, rolls, times, last_times, statuses, departments = extract_attendance()
    return render_template('dashboard.html', 
                           names=names, rolls=rolls, times=times, 
                           last_times=last_times, statuses=statuses, departments=departments)

@app.route('/attendance_history')
def attendance_history():
    files = glob.glob('Attendance/Attendance-*.csv')
    df_list = []

    for file in files:
        try:
            df = pd.read_csv(file, on_bad_lines='skip')
            # Extract date from filename: Attendance-YYYY-MM-DD.csv
            date_str = file.split('-')[-1].replace('.csv', '')
            df['Date'] = date_str
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if df_list:
        all_data = pd.concat(df_list, ignore_index=True)
    else:
        all_data = pd.DataFrame()

    # Get unique departments
    departments = all_data['Department'].unique() if not all_data.empty else []

    # Records by department (optional for detailed table)
    dept_records = {}
    for dept in departments:
        dept_records[dept] = all_data[all_data['Department'] == dept].to_dict(orient='records')

    # Count number of 'Present' per department
    attendance_counts = {}
    if not all_data.empty:
        attendance_counts = all_data.groupby('Department')['Status'].apply(lambda x: (x == 'Present').sum()).to_dict()

    # Prepare lists for Chart.js
    chart_labels = list(attendance_counts.keys())
    chart_values = list(attendance_counts.values())

    return render_template('attendance_history.html',
                           dept_records=dept_records,
                           attendance_counts=attendance_counts,
                           chart_labels=chart_labels,
                           chart_values=chart_values)

@app.route('/generate_report', methods=['GET', 'POST'])
def generate_report_route():
    if request.method == 'POST':
        email = request.form['email']
        message = send_email_report(email)
        flash(message)
        return redirect('/generate_report')
    return render_template('generate_report.html')

@app.route('/download_report', methods=['POST'])
def download_report():
    report_date = request.form.get('reportDate')
    try:
        report_date_obj = datetime.strptime(report_date, '%Y-%m-%d')
        formatted_date = report_date_obj.strftime('%m_%d_%y')
        report_filename = f'Attendance/Attendance-{formatted_date}.csv'
        return send_file(report_filename, as_attachment=True)
    except Exception as e:
        flash(f"Error generating report: {e}")
        return redirect('/generate_report')

@app.route('/mark_absent', methods=['POST'])
def mark_absent():
    mark_absent_users()
    return redirect(url_for('home_page'))

def get_user_department(name):
    # name is like "username_userid"
    user_folder = os.path.join('static', 'faces', name)
    info_path = os.path.join(user_folder, 'info.txt')
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            for line in f:
                if line.startswith("Department:"):
                    return line.split(":", 1)[1].strip()
    return "Unknown"

@app.route('/start', methods=['GET'])
def start():
    flash("Camera is now open. Please position yourself correctly.", "info")

    names, rolls, times, last_times, statuses, departments = extract_attendance()
    model_path = 'static/face_recognition_model.pkl'
    if not os.path.exists(model_path):
        return render_template('home.html', mess='No trained model found. Please add a user to continue.',
                               names=names, rolls=rolls, times=times, last_times=last_times,
                               statuses=statuses, departments=departments)

    try:
        with open(model_path, 'rb') as f:
            model = joblib.load(f)
    except Exception as e:
        return render_template('home.html', mess=f"Model file error: {e}")

    cap = cv2.VideoCapture(0)
    recognized_names = set()

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return render_template('home.html', mess="Failed to capture image from camera.",
                               names=names, rolls=rolls, times=times, last_times=last_times,
                               statuses=statuses, departments=departments)

    faces = extract_faces(frame)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (86, 32, 251), 2)
        face_img = cv2.resize(frame[y:y + h, x:x + w], (50, 50)).ravel().reshape(1, -1)

        try:
            name = model.predict(face_img)[0]
        except Exception:
            name = 'unknown'

        if name != 'unknown':
            if name not in recognized_names:
                recognized_names.add(name)
                add_attendance(name, "+916382845186")
                department = get_user_department(name)
                announce_attendance(f"Attendance recorded for {name} from {department} department")
            cv2.putText(frame, f'Welcome {name}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'Unknown', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return render_template('home.html', names=names, rolls=rolls, times=times, last_times=last_times,
                           statuses=statuses, departments=departments)
@app.route('/add', methods=['GET', 'POST'])
def add():
    if request.method == 'POST':
        newusername = request.form['newusername']
        newuserid = request.form['newuserid']
        department = request.form['department']

        # Create folder to store user face images
        userimagefolder = f'static/faces/{newusername}_{newuserid}'
        if not os.path.isdir(userimagefolder):
            os.makedirs(userimagefolder)

        # Save user details
        with open(os.path.join(userimagefolder, "info.txt"), "w") as f:
            f.write(f"Name: {newusername}\n")
            f.write(f"UserID: {newuserid}\n")
            f.write(f"Department: {department}\n")

        # Open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return "Error: Unable to access the camera"

        i, j = 0, 0  # Image counters
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            faces = extract_faces(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2)
                if j % 5 == 0:
                    img_path = os.path.join(userimagefolder, f"{i}.jpg")
                    cv2.imwrite(img_path, frame[y:y + h, x:x + w])
                    i += 1
                j += 1
            cv2.imshow('Face Capturing', frame)
            if i >= nimgs or cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        train_model()  # Retrain model with the new user data

        return render_template('home.html', mess="Face added successfully and model trained!")

    return render_template('add.html')  # Render the add user form



if __name__ == '__main__':
    app.run(debug=True)
