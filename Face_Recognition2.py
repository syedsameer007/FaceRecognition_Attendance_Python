import face_recognition
import numpy as np
import cv2 # opencv-python ka hai yeh
from datetime import datetime
import csv

Video_Capture = cv2.VideoCapture(0) #0 matlab ek web cam, 1 matlab dusra web cam,2 matlab teesra webcam

#Loading known faces
Sharukh_Image = face_recognition.load_image_file("Faces/Sharukh.jpg")
Sharukh_Encoding = face_recognition.face_encodings(Sharukh_Image)[0]#because yeh zero return karta hai, and agar ek sey zyada faces hai tho sif ek ko hi accept karega yeh #yeh sharuk ki image ko encoding bhanadeyga
Sameer_Image = face_recognition.load_image_file("Faces/Sameer.jpg")
Sameer_Encoding = face_recognition.face_encodings(Sameer_Image)[0]

known_face_encoding = [Sharukh_Encoding,Sameer_Encoding] #encoding jo bhanai hai unke naam store karenge
known_face_names = ["Sharukh Khan","Syed Sameer"] #names bhi agaye unke

#List of expected students
Students = known_face_names.copy() #jiske names hai unko copy kar raha hu 

face_locations = []  #yeh locations aur encoding hai issme
face_encodings = []

#get the current date and time, yeh issliye kiy ahai taki time ko lock karne ki kab insan aya hai
now =datetime.now()
current_date = now.strftime("%d-%m-%y")#yeh forbid karega, matlab order meh rakhega sub ko

#Ek CSV file bhanarahe hai, jo bhi date hogi hamari voh dot .csv file bhai jarahi hai
f = open(f"{current_date}.csv", "w+",newline="") #ek new file create kar rahe hai,aur usko write kar rahe hai 'w+', aur usko new line se none kar rahe hai
lnwriter = csv.writer(f,delimiter=',')

#yaha se magic hoga, while loop ko true karke unlimited times run karte rahe ga
while True:
    _, frame = Video_Capture.read()# why _?iska 1st argument yeh hai ki- kya apka phela videoCapture successful tha ki nahi, dusra argument frame hai, issliye usstaraha likha hai
    small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25) #yeh frame ko chota baada karne ke liye, hum tho chota kar rahe hai
    rgb_small_frame=cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)#yeh karne se BGR2RGB meh convert hojayega

    #abb hum recognize karenge faces ko
    face_locations = face_recognition.face_locations(rgb_small_frame)#yeh faces ko recognize karke unke bounding boxes ko return karega, tho yaha rgb_small_frames daldunga tab sare face yaha par miljayenge
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations) #isko bolrahe hai ki- ya ya faces hai iske encoding nikalo..abb inki encoding nikal gai hai, kiski encoding nikal gai hai? WEB CAM key dwara jo faces hai uski

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encoding, face_encoding)#yeh compare karega jo known faces hai ussey face encoding ke sath, and yeh True False meh return karega
        face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)#yeh yeh batayega ki kitna similar hai?known face aur face encoding ke beeech
        best_match_Index = np.argmin(face_distance) #yeh index batayega ki kitna hai...numpy ko use karke

        if(matches[best_match_Index]):# voh True False meh return karega, agar voh True hai tho andar ghuso
            name = known_face_names[best_match_Index]

        #adding a text if the person is present
        if name in known_face_names:
            font =cv2.FONT_HERSHEY_SIMPLEX
            BottomLeftConnerOfText = (10, 100)
            fontScale = 1.5 #kitna bada honna chaiye
            fontColor = (255, 0 ,0)
            thickness = 3
            lineType = 2
            cv2.putText(frame, name + "  Present", BottomLeftConnerOfText,font, fontScale, fontColor, thickness,lineType)

            if name in Students:
                Students.remove(name)
                current_time =now.strftime("%H %M %S") #hour, minuite, seconds print kararahe hai
                lnwriter.writerow([name, current_time])

    cv2.imshow("Attendance",frame) #frame show kar rahe hai
    if cv2.waitKey(1) & 0xFF ==ord("q"): #jab bhi q dabu yeh while loop exit hojaye
            break
    
Video_Capture.release() #yeh hamare camera ko free kardega
cv2.destroyAllWindows()#yeh sari windows destroy kardeytha hai
f.close()