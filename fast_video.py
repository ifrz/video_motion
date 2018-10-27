from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import datetime
import telegram
import threading

def notify(frame, now):
    if (time.time() - now < 1): # prevent from notify to many frames
        return
    bot = telegram.Bot(token='613818450:AAHD7sxmC9gJFQ_dulBt4Ajkxfk5JPHq5ZI')
    # print(bot.get_me())
    # curl https://api.telegram.org/bot613818450:AAHD7sxmC9gJFQ_dulBt4Ajkxfk5JPHq5ZI/getupdates
    cv2.imwrite("frame.jpg", frame)
    time.sleep(1) # to avoid send a corrupt image
    bot.send_message(chat_id='87534356', text="Somebody stole your beer!!")
    bot.send_photo(chat_id='87534356', photo=open('/home/pi/frame.jpg', 'rb'))

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
help="path to input video file")
args = vars(ap.parse_args())

# start the file video stream thread and allow the buffer to
# start to fill
print("[INFO] starting video file thread...")
fvs = FileVideoStream(args["video"]).start()
time.sleep(1.0)
bg = cv2.bgsegm.createBackgroundSubtractorMOG()

send = True
now = 0
# start the FPS timer
# fps = FPS().start()
# loop over frames from the video file stream
while (1):
# grab the frame from the threaded video file stream, resize
# it, and convert it to grayscale (while still retaining 3
# channels)
    frame = fvs.read()
    frame = imutils.resize(frame, width=450)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = np.dstack([frame, frame, frame])

    timestamp = datetime.datetime.now()
    text = "Unoccupied"

    motion = bg.apply(frame, learningRate=0.005)
    kernel = np.ones((3, 3), np.uint8)
    motion = cv2.morphologyEx(motion, cv2.MORPH_CLOSE, kernel, iterations=1)
    motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, kernel, iterations=1)
    motion = cv2.dilate(motion,kernel,iterations = 1)
    contours = cv2.findContours(motion, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = contours[0] if imutils.is_cv2() else contours[1]

    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < 5000:
            continue
        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Occupied"
        ### notify thread
        if (send == True and text is "Occupied"):
            t = threading.Thread(target=notify, args=(frame,now,))
            print ("[INFO] starting notify thread...")
            t.start()
            now = time.time()
        # draw the text and timestamp on the frame
        ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
        cv2.putText(frame, "Room Status: {}".format(text), (230, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
        0.35, (0, 0, 255), 1)

    # display the size of the queue on the frame
    cv2.putText(frame, "Queue Size: {}".format(fvs.Q.qsize()),
    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # show the frame and update the FPS counter
    cv2.imshow("Frame", frame)

    cv2.waitKey(1)

    # fps.update()

# stop the timer and display FPS information
#fps.stop()
#print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
#print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
fvs.stop()
