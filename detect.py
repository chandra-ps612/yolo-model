import cv2
import numpy as np
import argparse

'''This following object detection code works smoothly with yolo darknet model.'''

parser = argparse.ArgumentParser()
parser.add_argument('--webcam', help='True/False', default=False)
parser.add_argument('--video_feed', help='True/False', default=False)
parser.add_argument('--image', help='True/False', default=True)
parser.add_argument('--video_path', help='Path of video file', default='C:\\Users\\lav singh\\workplace\\yolo\\test.mp4')
parser.add_argument('--image_path', help='Path of image file', default='C:\\Users\\lav singh\\workplace\\yolo\\test.jpg')
parser.add_argument('--verbose', help='To print statements', default=True)
args = parser.parse_args()

configPath='C:\\Users\\lav singh\\workplace\\yolo\\yolov4-tiny.cfg'
weightsPath='C:\\Users\\lav singh\\workplace\\yolo\\yolov4-tiny.weights'
imagePath='C:\\Users\\lav singh\\workplace\\yolo\\test.jpg'
videoPath='C:\\Users\\lav singh\\workplace\\yolo\\test.mp4'
outputImagePath='C:\\Users\\lav singh\\workplace\\yolo\\imgr.jpg'

# Constants:
CONF_THRESHOLD=0.45
NMS_THRESHOLD=0.45
SCORE_THRESHOLD=0.5

# Load yolo pre-trained model
def load_yolo():
    net = cv2.dnn.readNet(configPath, weightsPath)
    # The list classes
    classes=[]
    with open('coco.names', 'r') as f:
        classes= [line.strip() for line in f.readlines()]
    layer_names=net.getLayerNames() # Get layers of the network
    # Determine the output layer names from the yolo pre-trained model
    output_layers=[layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers

def load_image(imagePath):
    img=cv2.imread(imagePath) # Loading image
    imgr=cv2.resize(img, (416, 416), fx=None, fy=None)
    height, width,_= imgr.shape
    return imgr, height, width

def start_webcam(): # Triggering webcam for realtime object detection
    cap=cv2.VideoCapture(0) # Here, '0' is to specify that uses built-in-camera
    return cap

def start_video(videoPath): #Triggering video source for object detection
    cap=cv2.VideoCapture(videoPath)
    return cap

def detect_objects(imgr, net, output_layers):
    # Using blob function of openCV to pre-process image
    blob=cv2.dnn.blobFromImage(imgr, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False) 
    net.setInput(blob)
    outputs=net.forward(output_layers)
    return blob, outputs

def get_box_dimensions(outputs, width, height): # Showing information on the screen
    class_ids=[]
    confs=[]
    boxes=[]
    for output in outputs:
        for detection in output:
            classes_scores= detection[5:]
            class_id=np.argmax(classes_scores)
            conf= detection[4]
            if conf>=CONF_THRESHOLD:
                center_x=int(detection[0]*width)
                center_y=int(detection[1]*height)
                w=int(detection[2]*width)
                h=int(detection[3]*height)
                # Rectangle co-ordinates
                x=int(center_x-w/2)
                y=int(center_y-h/2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return class_ids, confs, boxes

def draw_labels(boxes, confs, class_ids, classes, imgr):
    # Using NMS function of openCV to perform non-maximum suppression 
    indices=cv2.dnn.NMSBoxes(boxes, confs, SCORE_THRESHOLD, NMS_THRESHOLD)
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    for i in range (len(boxes)):
        if i in indices:
            x, y, w, h=boxes[i]
            conf=confs[i]
            class_id=class_ids[i]
            color = colors[class_ids[i]]
            cv2.rectangle(imgr, (x, y), (x+w, y+h), color, thickness=1)
            cv2.rectangle(imgr, (x,y), (x+w, y-25), color, thickness=-1)
            cv2.putText(imgr, f'{classes[class_id]}: {int(conf*100)}%', (x, y-5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1)
    cv2.imshow('output', imgr)


def image_detection(imagePath):
    net, classes, output_layers= load_yolo()
    imgr, height, width= load_image(imagePath)
    blob, outputs= detect_objects(imgr, net, output_layers)
    class_ids, confs, boxes= get_box_dimensions(outputs, width, height)
    draw_labels(boxes, confs, class_ids, classes, imgr)
    cv2.imwrite(outputImagePath, imgr)
    while True:
        key=cv2.waitKey(1)
        if key==ord('q'):
            break

def webcam_detection():
    net, classes, output_layers= load_yolo()
    cap= start_webcam()
    while True:
        _, frame=cap.read()
        frame=cv2.resize(frame, (416, 416), fx=None, fy=None)
        height, width,_= frame.shape
        blob, outputs= detect_objects(frame, net, output_layers)
        class_ids, confs, boxes= get_box_dimensions(outputs, height, width)
        draw_labels(boxes, confs, class_ids, classes, frame)
        key=cv2.waitKey(1)
        if key==ord('q'):
            break
    cap.release()

def video_detection(videoPath):
    create=None
    net, classes, output_layers= load_yolo()
    cap= start_video(videoPath)
    while True:
        _, frame=cap.read()
        frame=cv2.resize(frame, (416, 416), fx=None, fy=None)
        height, width,_= frame.shape
        blob, outputs= detect_objects(frame, net, output_layers)
        class_ids, confs, boxes= get_box_dimensions(outputs, height, width)
        draw_labels(boxes, confs, class_ids, classes, frame)
        if create is None:
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            create = cv2.VideoWriter(output_videoPath, fourcc, 10, (1920, 1080), True)
        create.write(frame)
        key=cv2.waitKey(1) # It'll generate a new frame after every 1 ms.
        if key==ord('q'):
            break
    cap.release()
    create.release()

if __name__ == '__main__':
	webcam = args.webcam
	video_feed = args.video_feed
	image = args.image
	if webcam:
		if args.verbose:
			print('---- Starting Webcam object detection ----')
		webcam_detection()
	if video_feed:
		videoPath = args.video_path
		if args.verbose:
			print('Opening '+videoPath+" .... ")
		start_video(videoPath)
	if image:
		imagePath = args.image_path
		if args.verbose:
			print("Opening "+imagePath+" .... ")
		image_detection(imagePath)
	cv2.destroyAllWindows()