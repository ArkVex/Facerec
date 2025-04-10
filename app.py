import cv2
import os
from flask import Flask, request, render_template
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import time

# Defining Flask App
app = Flask(__name__)

nimgs = 30

# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')

# get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))

# extract the face from an image
def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []

# Modify identify_face function to return prediction and confidence
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    pred = model.predict(facearray)
    prob = model.predict_proba(facearray).max()  # Get highest probability
    return pred[0], prob

# A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

# Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l

# Add Attendance of a specific user
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
        f.write(f'\n{username},{userid},{current_time}')

## A function to get names and rol numbers of all users
def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    l = len(userlist)

    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)

    return userlist, names, rolls, l

## A function to delete a user folder 
def deletefolder(duser):
    pics = os.listdir(duser)
    for i in pics:
        os.remove(duser+'/'+i)
    os.rmdir(duser)

# Function to delete an attendance record
def delete_attendance(username, userid, time):
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    # Find and remove the specific record
    df = df[~((df['Name'] == username) & (df['Roll'] == int(userid)) & (df['Time'] == time))]
    # Save the updated CSV
    df.to_csv(f'Attendance/Attendance-{datetoday}.csv', index=False)

################################ ROUTING FUNCTIONS #####################################

# Our main page
@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

## List users page
@app.route('/listusers')
def listusers():
    userlist, names, rolls, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)

## Delete functionality
@app.route('/deleteuser', methods=['GET'])
def deleteuser():
    duser = request.args.get('user')
    deletefolder('static/faces/'+duser)

    ## if all the face are deleted, delete the trained file...
    if os.listdir('static/faces/')==[]:
        os.remove('static/face_recognition_model.pkl')
    
    try:
        train_model()
    except:
        pass

    userlist, names, rolls, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)

# Add this new route for deleting attendance records
@app.route('/delete_attendance', methods=['GET'])
def delete_attendance_route():
    username = request.args.get('name')
    userid = request.args.get('roll')
    time = request.args.get('time')
    
    delete_attendance(username, userid, time)
    
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, 
                         totalreg=totalreg(), datetoday2=datetoday2)

# Our main Face Recognition functionality. 
# This function will run when we click on Take Attendance Button.
@app.route('/start', methods=['GET'])
def start():
    names, rolls, times, l = extract_attendance()

    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), 
                             datetoday2=datetoday2, mess='There is no trained model in the static folder. Please add a new face to continue.')

    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Could not access the camera")

        marked_attendance = set()
        start_time = time.time()
        last_recognition_time = 0
        recognition_cooldown = 2
        confidence_threshold = 0.6  # Set minimum confidence threshold

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            if time.time() - start_time > 10 and not marked_attendance:
                break

            faces = extract_faces(frame)
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
                cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
                face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
                identified_person, confidence = identify_face(face.reshape(1, -1))
                
                current_time = time.time()
                if current_time - last_recognition_time >= recognition_cooldown:
                    if confidence >= confidence_threshold:  # Only mark attendance if confidence is high
                        if identified_person not in marked_attendance:
                            add_attendance(identified_person)
                            marked_attendance.add(identified_person)
                            message = f'{identified_person} ({confidence:.2%})'
                        else:
                            message = f'{identified_person} (Already Marked)'
                    else:
                        message = 'Unknown Person'
                    last_recognition_time = current_time
                
                cv2.putText(frame, message, (x+5, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow('Attendance', frame)
            if cv2.waitKey(1) == 27:
                break

    except Exception as e:
        print(f"Camera error: {str(e)}")
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                             datetoday2=datetoday2, mess='Failed to access the camera. Please check if it is connected properly.')
    finally:
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()

    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Could not access the camera")

        i, j = 0, 0
        while i < nimgs:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            faces = extract_faces(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
                cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
                if j % 5 == 0:
                    name = newusername+'_'+str(i)+'.jpg'
                    cv2.imwrite(userimagefolder+'/'+name, frame[y:y+h, x:x+w])
                    i += 1
                j += 1

            cv2.imshow('Adding new User', frame)
            if cv2.waitKey(1) == 27:
                break

    except Exception as e:
        print(f"Camera error: {str(e)}")
        return render_template('home.html', names=[], rolls=[], times=[], l=0, totalreg=totalreg(),
                             datetoday2=datetoday2, mess='Failed to access the camera. Please check if it is connected properly.')
    finally:
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()

    print('Training Model')
    train_model()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

# Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)
