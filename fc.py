import cv2
import face_recognition
import numpy as np
import tkinter as tk
from tkinter import simpledialog, messagebox
import pyodbc
import pickle
import datetime

# SQL Server Connection (Windows Auth)
conn = pyodbc.connect(
    'Driver={SQL Server};'
    'Server=UNEEB'
    'Database=FaceSurveillance;'
    'Trusted_Connection=yes;'
)
cursor = conn.cursor()

# Insert new person into RegisteredUsers
def insert_registered_user(name, cnic, encoding):
    binary_encoding = pickle.dumps(encoding)
    cursor.execute("""
        INSERT INTO RegisteredUsers (Name, CNIC, Encoding) 
        VALUES (?, ?, ?)
    """, (name, cnic, binary_encoding))
    conn.commit()

# Load all registered face encodings
def load_registered_users():
    cursor.execute("SELECT Name, CNIC, Encoding FROM RegisteredUsers")
    rows = cursor.fetchall()
    encodings = []
    tags = []
    for name, cnic, bin_encoding in rows:
        enc = pickle.loads(bin_encoding)
        encodings.append(enc)
        tags.append(f"{name} - {cnic}")
    return encodings, tags

# Log surveillance detection
def log_detection(name, cnic):
    cursor.execute("""
        INSERT INTO SurveillanceLog (Name, CNIC, DetectedAt)
        VALUES (?, ?, ?)
    """, (name, cnic, datetime.datetime.now()))
    conn.commit()

# GUI Class
class FaceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Surveillance (SQL Server)")
        self.root.geometry("300x150")
        tk.Button(root, text="Register New Person", command=self.register_person, height=2, width=25).pack(pady=10)
        tk.Button(root, text="Start Live Surveillance", command=self.start_surveillance, height=2, width=25).pack()

    def register_person(self):
        name = simpledialog.askstring("Input", "Enter Name:")
        cnic = simpledialog.askstring("Input", "Enter CNIC:")
        if not name or not cnic:
            messagebox.showerror("Error", "Name and CNIC required")
            return

        cap = cv2.VideoCapture(0)
        messagebox.showinfo("Instructions", "Press 's' to capture face.")
        while True:
            ret, frame = cap.read()
            cv2.imshow("Registration - Press 's' to capture", frame)
            key = cv2.waitKey(1)
            if key == ord('s'):
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = face_recognition.face_locations(rgb)
                if faces:
                    encoding = face_recognition.face_encodings(rgb, faces)[0]
                    insert_registered_user(name, cnic, encoding)
                    messagebox.showinfo("Success", "Person registered successfully.")
                    break
                else:
                    messagebox.showerror("Error", "No face detected!")
        cap.release()
        cv2.destroyAllWindows()

    def start_surveillance(self):
        known_encodings, known_tags = load_registered_users()
        if not known_encodings:
            messagebox.showerror("Error", "No registered faces in database.")
            return

        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            faces = face_recognition.face_locations(rgb_small)
            encodings = face_recognition.face_encodings(rgb_small, faces)

            for face_loc, enc in zip(faces, encodings):
                matches = face_recognition.compare_faces(known_encodings, enc)
                face_dist = face_recognition.face_distance(known_encodings, enc)
                tag = "Unknown"
                if True in matches:
                    best_idx = np.argmin(face_dist)
                    tag = known_tags[best_idx]
                    name, cnic = tag.split(" - ")
                    log_detection(name, cnic)
                top, right, bottom, left = [v * 4 for v in face_loc]
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, tag, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            cv2.imshow("Live Surveillance - Press 'q' to exit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

# Launch App
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceApp(root)
    root.mainloop()
