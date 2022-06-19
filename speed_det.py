import cv2
import dlib
import time
import math
import imutils
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import easyocr

#import matplotlib.pyplot as plt

reader = easyocr.Reader(['en'])
carCascade = cv2.CascadeClassifier('cars.xml')
video = cv2.VideoCapture("runDone.mp4")
WIDTH = 1280
HEIGHT = 720

def anpr(car):
    print("ANPR Started")
        
    gray = cv2.cvtColor(car, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.GaussianBlur(gray, (9, 9), 20)
    edged = cv2.Canny(gray, 70, 110)
    cv2.imshow("cpImg",edged)
    
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = imutils.grab_contours(keypoints)
    cnts = sorted(cnt, key = cv2.contourArea, reverse = True)
    print("ANPR middle")
    location = None
    for contour in cnts:
        print(location)
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break
    mask = np.zeros(gray.shape, np.uint8)
    newImg = cv2.drawContours(mask, [location],0, 255, -1)
    
   
    newImg = cv2.bitwise_and(car,car, mask = mask)
    cv2.imshow("Img",newImg)
    print("ANPR middle next")
    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    
    print(x1, y1)
    
    cpImg = gray[x1: x2+1, y1: y2+1]
    
    cpImg = cv2.resize(cpImg, (200, 200))
    cv2.imshow("Cropped",cpImg)
    ce = cv2.Canny(cpImg, 25, 35)
    cv2.imshow("ce",ce)
    result = reader.readtext(newImg)
    
    
    """print("Result: ", result)
    if result == " " or result == None:
        return None
    else:
        return(result) 
    """
def estimateSpeed(location1, location2):
    d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    # ppm = location2[2] / carWidht
    ppm = 8.8
    d_meters = d_pixels / ppm
    fps = 60
    speed = d_meters * fps * 3.6
    return speed

def trackMultipleObjects():
    rectangleColor = (0, 255, 0)
    frameCounter = 0
    currentCarID = 0
    fps = 0

    carTracker = {}
    #carNumbers = {}
    carLocation1 = {}
    carLocation2 = {}
    speed = [None] * 1000
    anprDetected = [None] * 100

    out = cv2.VideoWriter('outNew.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (WIDTH, HEIGHT))

    while True:
        start_time = time.time()
        rc, image = video.read()
        if type(image) == type(None):
            break

        image = cv2.resize(image, (WIDTH, HEIGHT))
        resultImage = image.copy()

        frameCounter = frameCounter + 1
        carIDtoDelete = []

        for carID in carTracker.keys():
            trackingQuality = carTracker[carID].update(image)

            if trackingQuality < 10:
                carIDtoDelete.append(carID)

        
        for carID in carIDtoDelete:
            print("Removing carID " + str(carID) + ' from list of trackers. ')
            print("Removing carID " + str(carID) + ' previous location. ')
            print("Removing carID " + str(carID) + ' current location. ')
            carTracker.pop(carID, None)
            carLocation1.pop(carID, None)
            carLocation2.pop(carID, None)

        
        if not (frameCounter % 10):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cars = carCascade.detectMultiScale(gray, 1.1, 13, 18, (24, 24))

            for (_x, _y, _w, _h) in cars:
                x = int(_x)
                y = int(_y)
                w = int(_w)
                h = int(_h)

                x_bar = x + 0.5 * w
                y_bar = y + 0.5 * h

                matchCarID = None

                for carID in carTracker.keys():
                    trackedPosition = carTracker[carID].get_position()

                    t_x = int(trackedPosition.left())
                    t_y = int(trackedPosition.top())
                    t_w = int(trackedPosition.width())
                    t_h = int(trackedPosition.height())

                    t_x_bar = t_x + 0.5 * t_w
                    t_y_bar = t_y + 0.5 * t_h

                    if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and (x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
                        matchCarID = carID

                if matchCarID is None:
                    print(' Creating new tracker' + str(currentCarID))

                    tracker = dlib.correlation_tracker()
                    tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))

                    carTracker[currentCarID] = tracker
                    carLocation1[currentCarID] = [x, y, w, h]

                    currentCarID = currentCarID + 1

        for carID in carTracker.keys():
            trackedPosition = carTracker[carID].get_position()

            t_x = int(trackedPosition.left())
            t_y = int(trackedPosition.top())
            t_w = int(trackedPosition.width())
            t_h = int(trackedPosition.height())

            cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangleColor, 4)

            carLocation2[carID] = [t_x, t_y, t_w, t_h]
            carImage = resultImage[t_x: t_x + t_w, t_y: t_y + t_h]
            #print(anprDetected)
            #print(carImage)
            if speed[carID] is not None:
                if speed[carID] > 20:
                    try:
                        carImage = cv2.resize(carImage, (500, 500))
                        anpr(carImage)
                        
                    except:
                        pass

        end_time = time.time()

        if not (end_time == start_time):
            fps = 1.0/(end_time - start_time)

        for i in carLocation1.keys():
            if frameCounter % 1 == 0:
                [x1, y1, w1, h1] = carLocation1[i]
                [x2, y2, w2, h2] = carLocation2[i]

                carLocation1[i] = [x2, y2, w2, h2]

                if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                    if (speed[i] == None or speed[i] == 0): #and y1 >= 275 and y1 <= 285:
                        speed[i] = estimateSpeed([x1, y1, w1, h1], [x2, y2, w2, h2])
                        cv2.putText(resultImage, str(int(speed[i])) + "km/h", (int(x1 + w1/2), int(y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 100) ,2)
                        print("speed = " + str(speed[i]))           
                    
                    if speed[i] != None and y1 >= 180:
                        cv2.putText(resultImage, str(int(speed[i])) + "km/h", (int(x1 + w1/2), int(y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 100) ,2)

        

        out.write(resultImage)
        cv2.imshow(" result", resultImage)
        if cv2.waitKey(1) == 27:
            break

        #time.sleep(0.5)
    cv2.destroyAllWindows()
    out.release()

if __name__ == '__main__':
    trackMultipleObjects()
