import os
import cv2
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, date
from flask import (
    Flask, request, render_template, redirect,
    url_for, session, send_file, flash
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from twilio.rest import Client
import glob

# ---------------------------
# App & basic configuration
# ---------------------------
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change_me_in_prod")

# Paths & constants
ATT_DIR = "Attendance"
FACES_DIR = "static/faces"
MODEL_PATH = "static/face_recognition_model.pkl"
N_IMGS_PER_USER = 10

# Dates
def today_mm_dd_yy():
    return date.today().strftime("%m_%d_%y")

def today_human():
    return date.today().strftime("%d-%B-%Y")

# Ensure folders exist
os.makedirs(ATT_DIR, exist_ok=True)
os.makedirs(FACES_DIR, exist_ok=True)

# ---------------------------
# Twilio (optional)
# ---------------------------
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "").strip()

def twilio_enabled():
    return all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER])

def send_sms_notification(phone_e164: str, username: str, when: datetime):
    """Send SMS if Twilio is configured; else no-op."""
    try:
        if not (twilio_enabled() and phone_e164 and phone_e164.startswith("+")):
            return
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        body = f"Hello {username}, your attendance was recorded at {when.strftime('%H:%M:%S')}."
        client.messages.create(body=body, from_=TWILIO_PHONE_NUMBER, to=phone_e164)
    except Exception as e:
        # Don't crash the app if SMS fails
        print(f"[WARN] SMS send failed: {e}")

# ---------------------------
# CSV helpers
# ---------------------------
HEADER = "Name,Roll,Time,LastAttendanceTime,Status,Department\n"

def ensure_today_csv() -> str:
    path = os.path.join(ATT_DIR, f"Attendance-{today_mm_dd_yy()}.csv")
    if not os.path.isfile(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write(HEADER)
    return path

def get_latest_attendance_file() -> str | None:
    files = [f for f in os.listdir(ATT_DIR) if f.startswith("Attendance-") and f.endswith(".csv")]
    if not files:
        return None
    # Attendance-MM_DD_YY.csv
    def key(fn):
        try:
            mm, dd, yy = fn.replace("Attendance-", "").replace(".csv", "").split("_")
            return (int(yy), int(mm), int(dd))
        except Exception:
            return (0, 0, 0)
    files.sort(key=key, reverse=True)
    return os.path.join(ATT_DIR, files[0])

def extract_attendance():
    """Return lists for templating: names, rolls, times, last_times, statuses, departments."""
    csv_path = get_latest_attendance_file()
    if not csv_path or not os.path.exists(csv_path):
        print("[INFO] No attendance CSV found yet.")
        return [], [], [], [], [], []

    try:
        df = pd.read_csv(csv_path, on_bad_lines="skip")
        cols = df.columns.tolist()
        # Create missing columns if needed
        for col in ["Name", "Roll", "Time", "LastAttendanceTime", "Status", "Department"]:
            if col not in cols:
                df[col] = []
        return (
            df["Name"].fillna("").tolist(),
            df["Roll"].fillna("").tolist(),
            df["Time"].fillna("").tolist(),
            df["LastAttendanceTime"].fillna("").tolist(),
            df["Status"].fillna("").tolist(),
            df["Department"].fillna("").tolist(),
        )
    except Exception as e:
        print(f"[ERROR] Reading attendance file failed: {e}")
        return [], [], [], [], [], []

# ---------------------------
# Face utils
# ---------------------------
HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_CASCADE = cv2.CascadeClassifier(HAAR_PATH)

def extract_faces(frame_bgr):
    """Always return a Python list of (x, y, w, h) rectangles."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    rects = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    # OpenCV may return a tuple () or a NumPy array of shape (N,4)
    if isinstance(rects, tuple):
        return []
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in rects]

def total_registered_users():
    return len([d for d in os.listdir(FACES_DIR) if os.path.isdir(os.path.join(FACES_DIR, d))])

# ---------------------------
# Model train & predict (KNN)
# ---------------------------
def train_model():
    X, y = [], []
    labels_str = []
    for folder in os.listdir(FACES_DIR):
        folder_path = os.path.join(FACES_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        # folder is "username_userid"
        img_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if not img_files:
            continue
        for fn in img_files:
            img_path = os.path.join(folder_path, fn)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            face50 = cv2.resize(img, (50, 50)).reshape(-1)  # 2500 features
            X.append(face50)
            labels_str.append(folder)

    if not X:
        print("[ERROR] No training images found.")
        return False

    X = np.array(X, dtype=np.uint8)
    le = LabelEncoder()
    y = le.fit_transform(labels_str)

    clf = KNeighborsClassifier(n_neighbors=3, weights="distance")
    clf.fit(X, y)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"clf": clf, "le": le}, MODEL_PATH)
    print(f"[INFO] Model trained on {len(np.unique(y))} users / {len(X)} images.")
    return True

def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        bundle = joblib.load(MODEL_PATH)
        return bundle
    except Exception as e:
        print(f"[ERROR] Loading model failed: {e}")
        return None

def predict_name(face_bgr, model_bundle, prob_threshold=0.6):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    face50 = cv2.resize(gray, (50, 50)).reshape(1, -1)
    clf, le = model_bundle["clf"], model_bundle["le"]
    try:
        probs = clf.predict_proba(face50)[0]
        # clf.classes_ contains the class labels corresponding to the probability array
        argmax_idx = int(np.argmax(probs))
        pred_label = clf.classes_[argmax_idx]
        pmax = float(probs[argmax_idx])
        # Debugging hint (prints to server console)
        print(f"[DEBUG] predict_name probs={probs} argmax_idx={argmax_idx} pred_label={pred_label} pmax={pmax}")
        if pmax < prob_threshold:
            return "unknown"
        return le.inverse_transform([int(pred_label)])[0]
    except Exception:
        # Some sklearn versions or classifiers might not support predict_proba; fallback to predict
        try:
            pred_label = int(clf.predict(face50)[0])
            print(f"[DEBUG] predict_name fallback pred_label={pred_label}")
            return le.inverse_transform([pred_label])[0]
        except Exception:
            return "unknown"

# ---------------------------
# Pending approvals (in-memory)
# ---------------------------
pending_attendance = []

def add_attendance_pending(user_folder_name: str, phone_e164: str | None = None):
    """Add a recognized user to pending list (deduped by hour)."""
    now = datetime.now()

    try:
        username, userid = user_folder_name.split("_", 1)
    except ValueError:
        print(f"[WARN] Bad user folder format: {user_folder_name}")
        return

    # Prevent duplicate attendance within 1 hour
    for r in pending_attendance:
        if r["Name"] == username and r["Roll"] == userid:
            last = datetime.strptime(r["LastAttendanceTime"], "%Y-%m-%d %H:%M:%S")
            if (now - last).seconds < 3600:
                return

    # -------------------------
    # Read Department from info.txt
    # -------------------------
    department = "Unknown"

    info_path = os.path.join(FACES_DIR, user_folder_name, "info.txt")

    if os.path.exists(info_path):
        try:
            with open(info_path, "r", encoding="utf-8") as f:
                for line in f:
                    if "Department:" in line:
                        department = line.split(":", 1)[1].strip()
        except Exception as e:
            print(f"[WARN] Could not read department from {info_path}: {e}")

    # -------------------------
    # Create attendance record
    # -------------------------
    rec = {
        "Name": username,
        "Roll": userid,
        "Time": now.strftime("%H:%M:%S"),
        "LastAttendanceTime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "Status": "Present",
        "Department": department
    }

    pending_attendance.append(rec)

    # Send SMS if configured
    send_sms_notification(phone_e164 or "", username, now)
# ---------------------------
# Routes
# ---------------------------
@app.route("/")
def home_page():
    names, rolls, times, last_times, statuses, departments = extract_attendance()
    return render_template(
        "home.html",
        names=names, rolls=rolls, times=times,
        last_times=last_times, statuses=statuses, departments=departments
    )

@app.route("/admin_login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if username == "admin" and password == "admin123":
            session["admin_logged_in"] = True
            return redirect(url_for("admin_approval"))
        return render_template("admin_login.html", message="Invalid credentials.")
    return render_template("admin_login.html")

@app.route("/admin_approval", methods=["GET", "POST"])
def admin_approval():
    global pending_attendance

    if not session.get("admin_logged_in"):
        return redirect(url_for("admin_login"))

    if request.method == "POST":

        action = request.form.get("action")
        selected_rolls = request.form.getlist("attendance[]")
        valid = [r for r in selected_rolls if r]

        if action == "approve" and valid:

            try:
                csv_path = ensure_today_csv()

                with open(csv_path, "a", newline="", encoding="utf-8") as f:

                    for roll in valid:

                        rec = next((r for r in pending_attendance if r["Roll"] == roll), None)

                        if rec:

                            name = rec["Name"]
                            rec["Department"] = "unknown"

                            # Correct folder path
                            user_folder = f"{name}_{roll}"

                            info_path = os.path.join("D:\\face_recognition_flask-main\\static\\faces", user_folder, "info.txt")

                            if os.path.exists(info_path):

                                with open(info_path, "r") as file:
                                    lines = file.readlines()

                                    for line in lines:
                                        if "Department:" in line:
                                            department = line.split(":")[1].strip()

                            rec["Department"] = department

                            f.write(
                                f"{rec['Name']},{rec['Roll']},{rec['Time']},{rec['LastAttendanceTime']},{rec['Status']},{department}\n"
                            )

                    pending_attendance = [
                        r for r in pending_attendance if r["Roll"] not in valid
                    ]

                flash("Attendance approved.", "success")

            except Exception as e:
                flash(f"Error approving attendance: {e}", "danger")

        elif action == "reject":

            pending_attendance = [
                r for r in pending_attendance if r["Roll"] not in valid
            ]

            flash("Attendance rejected.", "warning")

    return render_template(
        "admin_approval.html",
        pending_attendance=pending_attendance,
        pending_count=len(pending_attendance)
    )
    
@app.route("/home")
def home():
    return render_template("home.html")


@app.route("/attendance_history")
def attendance_history():

    files = glob.glob(os.path.join(ATT_DIR, "Attendance-*.csv"))
    df_list = []

    for file in files:
        try:
            df = pd.read_csv(file, on_bad_lines="skip")

            # Extract date from filename
            date_str = os.path.basename(file).replace("Attendance-", "").replace(".csv", "")
            df["Date"] = pd.to_datetime(date_str, format="%m_%d_%y")

            df_list.append(df)
        except Exception as e:
            print(f"[WARN] Could not read {file}: {e}")

    if not df_list:
        return render_template(
            "attendance_history.html",
            dept_records={},
            chart_labels=[],
            chart_values=[],
            present_count=0,
            absent_count=0,
            monthly_labels=[],
            monthly_values=[],
            low_attendance=[]
        )

    # Combine all CSVs
    all_data = pd.concat(df_list, ignore_index=True)

    # Ensure required columns exist
    for col in ["Department", "Status", "Roll"]:
        if col not in all_data.columns:
            all_data[col] = "Unknown"

    # ---------------------------
    # Department-wise Present Count
    # ---------------------------
    attendance_counts = all_data.groupby("Department")["Status"].apply(
        lambda s: (s == "Present").sum()
    ).to_dict()

    chart_labels = list(attendance_counts.keys())
    chart_values = list(attendance_counts.values())

    # ---------------------------
    # Present vs Absent Count
    # ---------------------------
    present_count = int((all_data["Status"] == "Present").sum())
    absent_count = int((all_data["Status"] == "Absent").sum())

    # ---------------------------
    # Monthly Trend
    # ---------------------------
    all_data["Month"] = all_data["Date"].dt.strftime("%Y-%m")

    monthly_trend = all_data.groupby("Month")["Status"].apply(
        lambda s: (s == "Present").sum()
    ).to_dict()

    monthly_labels = list(monthly_trend.keys())
    monthly_values = list(monthly_trend.values())

    # ---------------------------
    # Low Attendance Alert (<75%)
    # ---------------------------
    student_summary = all_data.groupby("Roll")["Status"].apply(
        lambda x: (x == "Present").sum() / len(x) * 100
    ).reset_index(name="AttendancePercentage")

    low_attendance = student_summary[
        student_summary["AttendancePercentage"] < 75
    ].to_dict(orient="records")

    # ---------------------------
    # Department Records for Table
    # ---------------------------
    departments = all_data["Department"].dropna().unique().tolist()

    dept_records = {
        d: all_data[all_data["Department"] == d].to_dict(orient="records")
        for d in departments
    }

    return render_template(
        "attendance_history.html",
        dept_records=dept_records,
        chart_labels=chart_labels,
        chart_values=chart_values,
        present_count=present_count,
        absent_count=absent_count,
        monthly_labels=monthly_labels,
        monthly_values=monthly_values,
        low_attendance=low_attendance
    )

@app.route("/generate_report", methods=["GET", "POST"])
def generate_report():
    if request.method == "GET":
        return render_template("generate_report.html")
    
    # POST request - download the report
    report_date = request.form.get("reportDate", "")
    try:
        dt = datetime.strptime(report_date, "%Y-%m-%d")
        fname = f"Attendance-{dt.strftime('%m_%d_%y')}.csv"
        path = os.path.join(ATT_DIR, fname)
        if not os.path.exists(path):
            flash("No report for that date.", "warning")
            return redirect(url_for("generate_report"))
        return send_file(path, as_attachment=True)
    except Exception as e:
        flash(f"Error generating report: {e}", "danger")
        return redirect(url_for("generate_report"))

@app.route('/download_report', methods=['POST'])
def download_report():

    report_date = request.form.get("startDate")
    department = request.form.get("department")

    date_obj = datetime.strptime(report_date,"%Y-%m-%d")

    # Skip Sunday
    if date_obj.weekday() == 6:
        return "Sunday is holiday"

    file_name = f"Attendance/Attendance-{date_obj.strftime('%m_%d_%y')}.csv"

    if not os.path.exists(file_name):
        present_df = pd.DataFrame(columns=["Name","Roll","Department","Time"])
    else:
        present_df = pd.read_csv(file_name)

    # Load all registered students
    students = pd.read_csv("StudentDetails/studentdetails.csv")

    report = []

    for _, student in students.iterrows():

        roll = student["Roll"]
        name = student["Name"]
        dept = student["Department"]

        if department != "ALL" and dept != department:
            continue

        if roll in present_df["Roll"].values:
            status = "Present"
        else:
            status = "Absent"

        report.append({
            "Name":name,
            "Roll":roll,
            "Department":dept,
            "Date":report_date,
            "Status":status
        })

    report_df = pd.DataFrame(report)

    file_path = "reports/report.csv"
    report_df.to_csv(file_path,index=False)

    return send_file(file_path,as_attachment=True)
@app.route("/mark_absent", methods=["POST"])
def mark_absent():
    # Simple filler: ensure file exists; real "absent marking" needs a roster list
    ensure_today_csv()
    flash("Absent marking requires a master roster; not changed.", "info")
    return redirect(url_for("home_page"))

@app.route("/student/<roll>")
def student_profile(roll):

    import pandas as pd

    files = glob.glob(os.path.join("Attendance","Attendance-*.csv"))
    records = []

    for file in files:
        df = pd.read_csv(file)
        df = df[df["Roll"].astype(str) == str(roll)]
        if not df.empty:
            records.append(df)

    if records:
        student_data = pd.concat(records)
    else:
        student_data = pd.DataFrame()

    if not student_data.empty:
        name = student_data.iloc[0]["Name"]
        dept = student_data.iloc[0]["Department"]

        total = len(student_data)
        present = len(student_data[student_data["Status"]=="Present"])

        percentage = round((present/total)*100,2) if total else 0
    else:
        name="Unknown"
        dept="Unknown"
        percentage=0

    # Find student photo
    image_path = None
    possible_path = f"static/faces/{name}_{roll}"

    if os.path.exists(possible_path):
        files = os.listdir(possible_path)
        if files:
            image_path = f"/static/faces/{name}_{roll}/{files[0]}"

    return render_template(
        "student_profile.html",
        name=name,
        roll=roll,
        dept=dept,
        percentage=percentage,
        image=image_path,
        records=student_data.to_dict(orient="records")
    )

@app.route("/start", methods=["GET"])
def start():
    flash("Camera is opening. Please face the camera.", "info")

    names, rolls, times, last_times, statuses, departments = extract_attendance()

    model_bundle = load_model()
    if model_bundle is None:
        return render_template("home.html",
                               mess="No trained model found. Please add a user first.",
                               names=names, rolls=rolls, times=times,
                               last_times=last_times, statuses=statuses, departments=departments)

    # Try default; then DirectShow fallback (Windows)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        return render_template("home.html",
                               mess="Failed to open camera. If on Windows, allow camera access in Privacy settings.",
                               names=names, rolls=rolls, times=times,
                               last_times=last_times, statuses=statuses, departments=departments)

    # Grab one frame
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return render_template("home.html",
                               mess="Failed to capture image from camera.",
                               names=names, rolls=rolls, times=times,
                               last_times=last_times, statuses=statuses, departments=departments)

    faces = extract_faces(frame)  # now always a list
    seen = set()
    for (x, y, w, h) in faces:
        face_crop = frame[y:y+h, x:x+w]
        predicted = predict_name(face_crop, model_bundle, prob_threshold=0.6)
        if predicted != "unknown" and predicted not in seen:
            seen.add(predicted)
            # Change the phone number below to your recipient in +91… format if you want SMS:
            add_attendance_pending(predicted, phone_e164=None)

    if len(faces) == 0:
        flash("No face detected. Try better lighting & look straight at the camera.", "warning")
    elif not seen:
        flash("Face detected but not recognized. Please register first.", "warning")
    else:
        flash(f"Detected: {', '.join(seen)} (Pending admin approval).", "success")

    # Re-render with latest lists
    names, rolls, times, last_times, statuses, departments = extract_attendance()
    return render_template("home.html",
                           names=names, rolls=rolls, times=times,
                           last_times=last_times, statuses=statuses, departments=departments)
    

@app.route("/add", methods=["GET", "POST"])
def add():
    if request.method == "POST":
        newusername = request.form.get("newusername", "").strip()
        newuserid = request.form.get("newuserid", "").strip()
        department = request.form.get("department", "").strip() or "Unknown"

        if not newusername or not newuserid:
            return render_template("add.html", mess="Name and Roll are required.")

        user_folder = os.path.join(FACES_DIR, f"{newusername}_{newuserid}")
        os.makedirs(user_folder, exist_ok=True)

        with open(os.path.join(user_folder, "info.txt"), "w", encoding="utf-8") as f:
            f.write(f"Name: {newusername}\nUserID: {newuserid}\nDepartment: {department}\n")

        # Open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            return render_template("add.html", mess="Unable to access camera. Check permissions.")

        i, j = 0, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            faces = extract_faces(frame)  # list
            for (x, y, w, h) in faces:
                if j % 5 == 0 and i < N_IMGS_PER_USER:
                    crop = frame[y:y+h, x:x+w]
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(os.path.join(user_folder, f"{i}.jpg"), gray)
                    i += 1
                j += 1
            # Stop automatically once enough images captured
            if i >= N_IMGS_PER_USER:
                break

        cap.release()
        cv2.destroyAllWindows()

        if train_model():
            return render_template("home.html", mess="User added and model trained successfully!")
        else:
            return render_template("home.html", mess="User added, but model training failed. Add more images and retry.")

    return render_template("add.html")


if __name__ == "__main__":
   
    ensure_today_csv()
    app.run(debug=True)
