import cv2
import numpy as np
import pickle, os, sqlite3, random

def init_create_folder_database():
	# create the folder and database if not exist
    if not os.path.exists("gestures"):
        os.mkdir("gestures")
    if not os.path.exists("gesture_db.db"):
        conn = sqlite3.connect("gesture_db.db")
        create_table_cmd = "CREATE TABLE gesture ( g_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE, g_name TEXT NOT NULL )"
        conn.execute(create_table_cmd)
        conn.commit()


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

def store_in_db(g_id, g_name):
    conn = sqlite3.connect("gesture_db.db")
    cmd = "INSERT INTO gesture (g_id, g_name) VALUES (%s, \'%s\')" % (g_id, g_name)
    try:
        conn.execute(cmd)
    except sqlite3.IntegrityError:
        choice = input("g_id already exists. Want to change the record? (y/n): ")
        if choice.lower() == 'y':
            cmd = "UPDATE gesture SET g_name = \'%s\' WHERE g_id = %s" % (g_name, g_id)
            conn.execute(cmd)
        else:
            print("Doing nothing.....")
            return
    conn.commit()
    


def store_images(g_id):
    total_pics=3000

    #skin color
    lowerBound=np.array([0, 58, 50])
    upperBound=np.array([30, 255, 255])

    cam=cv2.VideoCapture(0)
    kernelOpen=np.ones((5,5))
    kernelClose=np.ones((20,20))

    create_folder("gestures/"+str(g_id))
    pic_no=0
    flag_start_capturing = False
    frames=0

    while True:
        x, y, w, h = 300, 100, 300, 300
        image_x, image_y = 50, 50

        ret,img=cam.read()
        img = cv2.flip(img, 1)
        img=cv2.resize(img,(640,480))

        imgHSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        mask=cv2.inRange(imgHSV,lowerBound,upperBound)
        thresh = mask[y:y+h, x:x+w]
        contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
        if len(contours)>0:
            contour=max(contours,key=cv2.contourArea)
            if cv2.contourArea(contour)>10000 and frames > 50:
                
                x1,y1,w1,h1 = cv2.boundingRect(contour)
                pic_no += 1
                save_img = thresh[y1:y1+h1,x1:x1+w1]
                if w1>h1:
                    save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2) , int((w1-h1)/2) , 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0)) #input,top,bottom,left,right,border_type,value
                elif h1> w1:
                    save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2) , int((h1-w1)/2) , cv2.BORDER_CONSTANT, (0, 0, 0))
                save_img = cv2.resize(save_img, (image_x, image_y))
                rand = random.randint(0, 10)
                if rand % 2 == 0:
                    save_img = cv2.flip(save_img, 1)
                cv2.putText(img, "Capturing...", (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 2, (127, 255, 255))
                cv2.imwrite("gestures/"+str(g_id)+"/"+str(pic_no)+".jpg", save_img)
                #cv2.imwrite("gesture/"+str(pic_no)+".jpg",save_img)
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(img, str(pic_no), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))
        cv2.imshow('mask',thresh)
        cv2.imshow('img',img)
        keypress = cv2.waitKey(1)
        if keypress == ord('c'):
            if flag_start_capturing == False:
                flag_start_capturing = True
            else:
                flag_start_capturing = False
                frames=0
        if flag_start_capturing == True:
            frames +=1
        if pic_no == total_pics:
            break
        
        
init_create_folder_database()
g_id = input("Enter gesture no.: ")
g_name = input("Enter gesture name/text: ")
store_in_db(g_id, g_name)
store_images(g_id)