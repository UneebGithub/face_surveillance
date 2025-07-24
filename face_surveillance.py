import cv2
import face_recognition
import numpy as np
import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog
import pyodbc
import pickle
import datetime

# SQL Server Connection
conn = pyodbc.connect(
    'Driver={SQL Server};'
    'Server=UNEEB;'  # Change if your server name is different
    'Database=FaceSurveillance;'
    'Trusted_Connection=yes;'
)
cursor = conn.cursor()

# Create tables if they don't exist
cursor.execute("""
    IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'RegisteredUsers')
    CREATE TABLE RegisteredUsers (
        Id INT PRIMARY KEY IDENTITY,
        Name NVARCHAR(100),
        CNIC NVARCHAR(20),
        StudentRegNo NVARCHAR(20),
        Semester NVARCHAR(20),
        Encoding VARBINARY(MAX)
    )
""")
cursor.execute("""
    IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'SurveillanceLog')
    CREATE TABLE SurveillanceLog (
        Id INT PRIMARY KEY IDENTITY,
        Name NVARCHAR(100),
        CNIC NVARCHAR(20),
        StudentRegNo NVARCHAR(20),
        Semester NVARCHAR(20),
        DetectedAt DATETIME DEFAULT GETDATE()
    )
""")
cursor.execute("""
    IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'AttendanceLog')
    CREATE TABLE AttendanceLog (
        Id INT PRIMARY KEY IDENTITY,
        Name NVARCHAR(100),
        CNIC NVARCHAR(20),
        StudentRegNo NVARCHAR(20),
        Semester NVARCHAR(20),
        AttendanceTime DATETIME DEFAULT GETDATE()
    )
""")
conn.commit()

# Insert new person into RegisteredUsers
def insert_registered_user(name, cnic, reg_no, semester, encoding):
    binary_encoding = pickle.dumps(encoding)
    cursor.execute("""
        INSERT INTO RegisteredUsers (Name, CNIC, StudentRegNo, Semester, Encoding) 
        VALUES (?, ?, ?, ?, ?)
    """, (name, cnic, reg_no, semester, binary_encoding))
    conn.commit()

# Load all registered face encodings
def load_registered_users():
    cursor.execute("SELECT Id, Name, CNIC, StudentRegNo, Semester, Encoding FROM RegisteredUsers")
    rows = cursor.fetchall()
    encodings = []
    tags = []
    user_info = {}
    for id, name, cnic, reg_no, semester, bin_encoding in rows:
        enc = pickle.loads(bin_encoding)
        encodings.append(enc)
        tag = f"{name} - {cnic}"
        tags.append(tag)
        user_info[tag] = {
            'id': id,
            'name': name,
            'cnic': cnic,
            'reg_no': reg_no,
            'semester': semester
        }
    return encodings, tags, user_info

# Log surveillance detection
def log_detection(name, cnic, reg_no, semester):
    cursor.execute("""
        INSERT INTO SurveillanceLog (Name, CNIC, StudentRegNo, Semester, DetectedAt)
        VALUES (?, ?, ?, ?, GETDATE())
    """, (name, cnic, reg_no, semester))
    conn.commit()

# Log attendance
def log_attendance(name, cnic, reg_no, semester):
    cursor.execute("""
        INSERT INTO AttendanceLog (Name, CNIC, StudentRegNo, Semester, AttendanceTime)
        VALUES (?, ?, ?, ?, GETDATE())
    """, (name, cnic, reg_no, semester))
    conn.commit()

# GUI Application
class FaceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Surveillance (Multi-User)")
        self.root.geometry("300x200")
        tk.Button(root, text="Register New Person", command=self.register_person, height=2, width=25).pack(pady=10)
        tk.Button(root, text="Start Live Surveillance", command=self.start_surveillance, height=2, width=25).pack(pady=10)
        tk.Label(root, text="Press 'I' to show info, 'A' to mark attendance, 'Q' to quit").pack(pady=10)

    def register_person(self):
        name = simpledialog.askstring("Input", "Enter Name:")
        cnic = simpledialog.askstring("Input", "Enter CNIC:")
        reg_no = simpledialog.askstring("Input", "Enter Student Registration No:")
        semester = simpledialog.askstring("Input", "Enter Semester:")
        if not all([name, cnic, reg_no, semester]):
            messagebox.showerror("Error", "All fields are required.")
            return

        capture_method = messagebox.askquestion("Capture Method", "Do you want to capture live? (Yes = Live, No = Upload)")
        encoding = None

        if capture_method.lower() == 'yes':
            cap = cv2.VideoCapture(0)
            messagebox.showinfo("Instructions", "Press 's' to capture face.")
            while True:
                ret, frame = cap.read()
                cv2.imshow("Capture - Press 's'", frame)
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    faces = face_recognition.face_locations(rgb)
                    if faces:
                        encoding = face_recognition.face_encodings(rgb, faces)[0]
                        break
                    else:
                        messagebox.showerror("Error", "No face detected!")
            cap.release()
            cv2.destroyAllWindows()
        else:
            file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
            if file_path:
                image = face_recognition.load_image_file(file_path)
                faces = face_recognition.face_locations(image)
                if faces:
                    encoding = face_recognition.face_encodings(image, faces)[0]
                else:
                    messagebox.showerror("Error", "No face detected in uploaded image.")
                    return

        if encoding is not None:
            insert_registered_user(name, cnic, reg_no, semester, encoding)
            messagebox.showinfo("Success", "Person registered successfully.")

    import cv2
import face_recognition
import numpy as np
import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog
import pyodbc
import pickle
import datetime

# SQL Server Connection
conn = pyodbc.connect(
    'Driver={SQL Server};'
    'Server=UNEEB;'  # Change to your SQL Server name if needed
    'Database=FaceSurveillance;'
    'Trusted_Connection=yes;'
)
cursor = conn.cursor()

# Create tables if not exist
cursor.execute("""
    IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'RegisteredUsers')
    CREATE TABLE RegisteredUsers (
        Id INT PRIMARY KEY IDENTITY,
        Name NVARCHAR(100),
        CNIC NVARCHAR(20),
        StudentRegNo NVARCHAR(20),
        Semester NVARCHAR(20),
        Encoding VARBINARY(MAX)
    )
""")
cursor.execute("""
    IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'SurveillanceLog')
    CREATE TABLE SurveillanceLog (
        Id INT PRIMARY KEY IDENTITY,
        Name NVARCHAR(100),
        CNIC NVARCHAR(20),
        StudentRegNo NVARCHAR(20),
        Semester NVARCHAR(20),
        DetectedAt DATETIME DEFAULT GETDATE()
    )
""")
cursor.execute("""
    IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'AttendanceLog')
    CREATE TABLE AttendanceLog (
        Id INT PRIMARY KEY IDENTITY,
        Name NVARCHAR(100),
        CNIC NVARCHAR(20),
        StudentRegNo NVARCHAR(20),
        Semester NVARCHAR(20),
        AttendanceTime DATETIME DEFAULT GETDATE()
    )
""")
conn.commit()

# Insert new user
def insert_registered_user(name, cnic, reg_no, semester, encoding):
    binary_encoding = pickle.dumps(encoding)
    cursor.execute("""
        INSERT INTO RegisteredUsers (Name, CNIC, StudentRegNo, Semester, Encoding) 
        VALUES (?, ?, ?, ?, ?)
    """, (name, cnic, reg_no, semester, binary_encoding))
    conn.commit()

# Load users
def load_registered_users():
    cursor.execute("SELECT Id, Name, CNIC, StudentRegNo, Semester, Encoding FROM RegisteredUsers")
    rows = cursor.fetchall()
    encodings = []
    tags = []
    user_info = {}
    for id, name, cnic, reg_no, semester, bin_encoding in rows:
        enc = pickle.loads(bin_encoding)
        encodings.append(enc)
        tag = f"{name} - {cnic}"
        tags.append(tag)
        user_info[tag] = {
            'id': id,
            'name': name,
            'cnic': cnic,
            'reg_no': reg_no,
            'semester': semester
        }
    return encodings, tags, user_info

# Log detection
def log_detection(name, cnic, reg_no, semester):
    cursor.execute("""
        INSERT INTO SurveillanceLog (Name, CNIC, StudentRegNo, Semester, DetectedAt)
        VALUES (?, ?, ?, ?, GETDATE())
    """, (name, cnic, reg_no, semester))
    conn.commit()

# Log attendance
def log_attendance(name, cnic, reg_no, semester):
    cursor.execute("""
        INSERT INTO AttendanceLog (Name, CNIC, StudentRegNo, Semester, AttendanceTime)
        VALUES (?, ?, ?, ?, GETDATE())
    """, (name, cnic, reg_no, semester))
    conn.commit()

# GUI
class FaceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Surveillance (Auto-Attendance)")
        self.root.geometry("320x220")
        tk.Button(root, text="Register New Person", command=self.register_person, height=2, width=25).pack(pady=10)
        tk.Button(root, text="Start Live Surveillance", command=self.start_surveillance, height=2, width=25).pack(pady=10)
        tk.Label(root, text="System will mark attendance automatically.").pack(pady=10)
        tk.Label(root, text="Press 'I' to show info, 'Q' to quit.").pack()

    def register_person(self):
        name = simpledialog.askstring("Input", "Enter Name:")
        cnic = simpledialog.askstring("Input", "Enter CNIC:")
        reg_no = simpledialog.askstring("Input", "Enter Student Registration No:")
        semester = simpledialog.askstring("Input", "Enter Semester:")
        if not all([name, cnic, reg_no, semester]):
            messagebox.showerror("Error", "All fields are required.")
            return

        method = messagebox.askquestion("Capture Method", "Do you want to capture live? (Yes = Live, No = Upload)")
        encoding = None

        if method.lower() == 'yes':
            cap = cv2.VideoCapture(0)
            messagebox.showinfo("Instructions", "Press 's' to capture face.")
            while True:
                ret, frame = cap.read()
                cv2.imshow("Capture - Press 's'", frame)
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    faces = face_recognition.face_locations(rgb)
                    if faces:
                        encoding = face_recognition.face_encodings(rgb, faces)[0]
                        break
                    else:
                        messagebox.showerror("Error", "No face detected!")
            cap.release()
            cv2.destroyAllWindows()
        else:
            file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
            if file_path:
                image = face_recognition.load_image_file(file_path)
                faces = face_recognition.face_locations(image)
                if faces:
                    encoding = face_recognition.face_encodings(image, faces)[0]
                else:
                    messagebox.showerror("Error", "No face detected in uploaded image.")
                    return

        if encoding is not None:
            insert_registered_user(name, cnic, reg_no, semester, encoding)
            messagebox.showinfo("Success", "Person registered successfully.")

    def start_surveillance(self):
        known_encodings, known_tags, user_info = load_registered_users()
        if not known_encodings:
            messagebox.showerror("Error", "No registered users.")
            return

        cap = cv2.VideoCapture(0)
        trackers = []
        frame_count = 0
        shown_info = set()
        auto_marked_users = set()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if frame_count % 10 == 0:
                small = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)
                faces = face_recognition.face_locations(small)
                encodings = face_recognition.face_encodings(small, faces)

                for face_loc, enc in zip(faces, encodings):
                    matches = face_recognition.compare_faces(known_encodings, enc, tolerance=0.5)
                    distances = face_recognition.face_distance(known_encodings, enc)

                    tag = "Unknown"
                    info = None
                    if True in matches:
                        best = np.argmin(distances)
                        if distances[best] < 0.5:
                            tag = known_tags[best]
                            info = user_info[tag]

                    top, right, bottom, left = [v * 4 for v in face_loc]
                    box = (left, top, right - left, bottom - top)

                    if tag != "Unknown" and not any(t == tag for _, t, _ in trackers):
                        try:
                            tracker = cv2.TrackerCSRT_create()
                        except:
                            tracker = cv2.legacy.TrackerCSRT_create()
                        tracker.init(frame, box)
                        trackers.append((tracker, tag, info))

                        # ✅ Auto mark attendance
                        if info and info['cnic'] not in auto_marked_users:
                            log_attendance(info['name'], info['cnic'], info['reg_no'], info['semester'])
                            log_detection(info['name'], info['cnic'], info['reg_no'], info['semester'])
                            auto_marked_users.add(info['cnic'])
                            if info['cnic'] not in shown_info:
                                shown_info.add(info['cnic'])
                                messagebox.showinfo("Attendance Marked",
                                    f"Attendance marked for:\n\nName: {info['name']}\nCNIC: {info['cnic']}\nReg#: {info['reg_no']}\nSemester: {info['semester']}")

            updated = []
            for tracker, tag, info in trackers:
                success, box = tracker.update(frame)
                if success:
                    x, y, w, h = map(int, box)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, tag, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                    updated.append((tracker, tag, info))
            trackers = updated

            cv2.imshow("Surveillance - Auto Attendance | Press 'I' for Info | Q to Quit", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('i') and trackers:
                info_text = ""
                for _, tag, info in trackers:
                    if info:
                        info_text += f"Name: {info['name']}\nCNIC: {info['cnic']}\nReg#: {info['reg_no']}\nSemester: {info['semester']}\n\n"
                messagebox.showinfo("Detected Persons", info_text if info_text else "No known persons detected.")

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()

# Launch the app
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceApp(root)
    root.mainloop()
