from flask import Flask, request, render_template
import os
import face_recognition
import pickle
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Load the pretrained model (face encodings and student names)
def load_trained_model(model_file='trained_model.pickle'):
    with open(model_file, 'rb') as f:
        student_encodings, student_names = pickle.load(f)
    return student_encodings, student_names

# Mark attendance based on recognized faces
def mark_attendance(student_names, recognized_names):
    now = datetime.now()
    date = now.strftime('%d-%m-%Y')
    time = now.strftime('%H:%M')

    # Create attendance DataFrame
    attendance = pd.DataFrame(student_names, columns=['Student Name'])
    attendance['Date'] = date
    attendance['Time'] = time
    attendance['Status'] = ['Present' if name in recognized_names else 'Absent' for name in student_names]

    return attendance

# Recognize faces in a group photo
def recognize_faces_in_group_photo(group_photo_path, student_encodings, student_names):
    group_photo = face_recognition.load_image_file(group_photo_path)
    small_frame = cv2.resize(group_photo, (0, 0), fx=0.5, fy=0.5)
    face_locations = face_recognition.face_locations(small_frame)
    face_locations = [(top * 2, right * 2, bottom * 2, left * 2) for top, right, bottom, left in face_locations]
    face_encodings = face_recognition.face_encodings(group_photo, face_locations)

    recognized_names = []
    for encoding in face_encodings:
        matches = face_recognition.compare_faces(student_encodings, encoding)
        face_distances = face_recognition.face_distance(student_encodings, encoding)
        best_match_index = face_distances.argmin()

        if matches[best_match_index]:
            recognized_names.append(student_names[best_match_index])

    return recognized_names

# Main processing function
def process_attendance(group_photo_path):
    student_encodings, student_names = load_trained_model()
    recognized_names = recognize_faces_in_group_photo(group_photo_path, student_encodings, student_names)
    attendance = mark_attendance(student_names, recognized_names)
    attendance.to_csv('attendance.csv', index=False)
    return attendance

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['class_photo']
        if file:
            # Save the uploaded file
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)

            # Process attendance
            attendance = process_attendance(file_path)

            return render_template('attendance_results.html', attendance=attendance.to_dict(orient='records'))

    return render_template('attendance.html')

if __name__ == '__main__':
    app.run(debug=True)
