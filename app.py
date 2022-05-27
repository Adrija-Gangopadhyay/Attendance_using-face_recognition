from flask import Flask, render_template, Response, request
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pandas as pd
from flask_mail import Mail, Message


app = Flask(__name__)
camera = cv2.VideoCapture(0)

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'arenesnells@gmail.com'
app.config['MAIL_PASSWORD'] = 'Hagudada@20'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True

mail = Mail(app)


path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])



def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name}, {dtString}')


encodeListKnown = findEncodings(images)


def generate_frames():
    while True:
        success, frame = camera.read()

        if not success:
            break
        else:
            imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                faceDis = faceDis*100
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    name = classNames[matchIndex].upper()
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    markAttendance(name)

            def __init__(self):
                self.video = cv2.VideoCapture(0)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
            b'Content-Type:image/jpeg\r\n\r\n' + frame + b'\r\n')


df = pd.read_csv("Attendance.csv")
df.to_csv("Attendance.csv", index=None)


@app.route('/')
def index():
    return render_template('index2.html')


@app.route('/table')
def csvtohtml():
    data = pd.read_csv("Attendance.csv")
    return render_template("upload.html", tables=[data.to_html()], titles=[''])


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/message')
def index_message():
    return render_template("home.html")


@app.route('/send_message', methods=['GET', 'POST'])
def send_message():
    if request.method == "POST":
        email = request.form['email']
        subject = "Notification regarding attendance"
        msg = "Greetings! Attached below is the attendance sheet. Thank You!"
        message = Message(subject, sender="arenesnells@gmail.com", recipients=[email])

        message.body = msg
        with app.open_resource('Attendance.csv') as cat:
            message.attach('Attendance.csv', 'text/csv', cat.read())
        mail.send(message)

        success = "Message sent"

        return render_template("result.html", success=success)


if __name__  == "__main__":
    app.run(debug=True)