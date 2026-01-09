#Import the Open-CV extra functionalities
import cv2
import asyncio
from time import sleep
import speech_recognition as sr
# import RPi.GPIO as GPIO

# Left_Forward_Pin = 35 #the internal Pi pin number that goes to snap 1
# Left_Backward_Pin = 31 #the internal Pi pin number that goes to snap 2
# Right_Forward_Pin = 26 #the internal Pi pin number that goes to snap 3
# Right_Backward_Pin = 21 #the internal Pi pin number that goes to snap 4

# GPIO.setmode(GPIO.BOARD)
# #Our output pins, start off
# GPIO.setup(Left_Forward_Pin, GPIO.OUT, initial=GPIO.LOW)
# GPIO.setup(Left_Backward_Pin, GPIO.OUT, initial=GPIO.LOW)
# GPIO.setup(Right_Forward_Pin, GPIO.OUT, initial=GPIO.LOW)
# GPIO.setup(Right_Backward_Pin, GPIO.OUT, initial=GPIO.LOW)

# def drive_forward(time):
#   GPIO.output(Left_Forward_Pin, GPIO.HIGH) #Left motor forward
#   GPIO.output(Right_Forward_Pin, GPIO.HIGH) #Right motor forward
#   sleep(time)
#   GPIO.output(Left_Forward_Pin, GPIO.LOW) #Left motor off
#   GPIO.output(Right_Forward_Pin, GPIO.LOW) #Right motor off
#   print('forward')
  
# def drive_left_turn(time):
#   GPIO.output(Left_Backward_Pin, GPIO.HIGH) #Left motor backward
#   GPIO.output(Right_Forward_Pin, GPIO.HIGH) #Right motor forward
#   sleep(time)
#   GPIO.output(Left_Backward_Pin, GPIO.LOW) #Left motor off
#   GPIO.output(Right_Forward_Pin, GPIO.LOW) #Right motor off
#   print('left turn')
#   sleep(1)
  
# def drive_right_turn(time):
#   GPIO.output(Left_Forward_Pin, GPIO.HIGH) #Left motor forward
#   GPIO.output(Right_Backward_Pin, GPIO.HIGH) #Right motor backward
#   sleep(time)
#   GPIO.output(Left_Forward_Pin, GPIO.LOW) #Left motor off
#   GPIO.output(Right_Backward_Pin, GPIO.LOW) #Right motor off
#   print('right turn')
#   sleep(1)
  
# def drive_backward(time):
#   GPIO.output(Left_Backward_Pin, GPIO.HIGH) #Left motor backward
#   GPIO.output(Right_Backward_Pin, GPIO.HIGH) #Right motor backward
#   sleep(time)
#   GPIO.output(Left_Backward_Pin, GPIO.LOW) #Left motor off
#   GPIO.output(Right_Backward_Pin, GPIO.LOW) #Right motor off
#   print('backward')
#   sleep(1)


recognizer = sr.Recognizer()

lst = []

text = ""

# Use the microphone as the audio source
with sr.Microphone() as source:
    print("Adjusting for ambient noise... Please wait.")
    recognizer.adjust_for_ambient_noise(source, duration=1)
    print("Listening... Speak now!")

    try:
        # Capture the audio
        audio = recognizer.listen(source)
        print("Recognizing speech...")
        
        # Convert speech to text using Google Web Speech API
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand the audio.")
    except sr.RequestError as e:
        print(f"Could not request results; {e}")

if text.split(' ')[-1] == "person" or text.split(' ')[-1] == "people":
    lst.append("person")
elif text.split(' ')[-1] == "phone" or text.split(' ')[-1] == "phones":
    lst.append("cell phone")
elif text.split(' ')[-1] == "anything" or text.split(' ')[-1] == "everything":
    lst = []
else:
    lst.append(text.split(' ')[-1])
    # drive_forward(2)

#This is to pull the information about what each object is called
classNames = []
classFile = "Object_Detection_Folder\\Object_Detection_Files\\coco 1.names"
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

#This is to pull the information about what each object should look like
configPath = "Object_Detection_Folder\\Object_Detection_Files\\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "Object_Detection_Folder\\Object_Detection_Files\\frozen_inference_graph.pb"

#This is some set up values to get good results
net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

#This is to set up what the drawn box size/colour is and the font/size/colour of the name tag and confidence label   
def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
#Below has been commented out, if you want to print each sighting of an object to the console you can uncomment below     
    print(classIds,bbox)
    if len(objects) == 0: objects = classNames
    objectInfo =[]
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = classNames[classId - 1]
            if className in objects: 
                objectInfo.append([box,className])
                # GPIO.output(Left_Forward_Pin, GPIO.HIGH) #Left motor forward
                # GPIO.output(Right_Forward_Pin, GPIO.HIGH) #Right motor forward
                if (draw) and confidence > 0.6:
                    cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                    cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            # else:
            #     GPIO.output(Left_Forward_Pin, GPIO.LOW) #Left motor forward
            #     GPIO.output(Right_Forward_Pin, GPIO.LOW) #Right motor forward
    
    return img,objectInfo

async def capture_frame(cap, lst):
    while True:
        success, img = cap.read()
        if not success:
            break
        result, objectInfo = getObjects(img, 0.45, 0.2, objects=lst)
        cv2.imshow("Output", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Check for 'Q' key press
            break
        await asyncio.sleep(0.01)  # Add a small delay to allow other tasks to run

async def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    await capture_frame(cap, lst)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())

# GPIO.cleanup()
