import cv2
import asyncio
import numpy as np

# Load class names
classNames = []
classFile = "Object_Detection_Folder/Object_Detection_Files/coco 1.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

# Load model configuration and weights
configPath = "Object_Detection_Folder/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "Object_Detection_Folder/Object_Detection_Files/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)
    print(classIds, bbox)
    if len(objects) == 0: objects = classNames
    objectInfo = []
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box, className])
                if draw and confidence > 0.6:
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, className.upper(), (box[0]+10, box[1]+30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence*100, 2)), (box[0]+200, box[1]+30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    return img, objectInfo

async def capture_frame(pipe_path, lst):
    width, height = 640, 480
    with open(pipe_path, 'rb') as pipe:
        while True:
            yuv = pipe.read(width * height * 3 // 2)
            if not yuv:
                break
            yuv = np.frombuffer(yuv, dtype=np.uint8).reshape((height * 3 // 2, width))
            img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
            result, objectInfo = getObjects(img, 0.45, 0.2, objects=lst)
            cv2.imshow("Output", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            await asyncio.sleep(0.01)

async def main():
    pipe_path = '/tmp/vidpipe'
    lst = []  # Analyze all objects
    await capture_frame(pipe_path, lst)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())
