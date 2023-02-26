import math
import base64
import cv2
import numpy as np
import matplotlib.pyplot as plt

FONT_SCALE = 1e-3  # Adjust for larger font size in all images
THICKNESS_SCALE = 1e-3  # Adjust for larger thickness in all images
TEXT_Y_OFFSET_SCALE = 1e-2  # Adjust for larger Y-offset of text and bounding box

def model_loading():
    # reads neural network model
    # net = cv2.dnn.readNet("models\yolov3.weights", "models\yolov3.cfg")
    net = cv2.dnn.readNetFromDarknet("models/yolov3.cfg", "models/yolov3.weights")
    
    # reads all identifiable objects from coco.names    
    with open("models/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # getting indexes of all output layers
    layers_names = net.getLayerNames()
    out_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
    
    # returns neural network, object classes, output_layers of network
    return net, classes, out_layers

def image_proccesing(img_bytes):
    # reads and resizes image using scale factor of 0.7
    file_bytes = np.fromstring(img_bytes, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    img = cv2.resize(img, None, fx=0.7, fy=0.7)
    height, width, channels = img.shape
    
    # returns image, height, width, channels(RBG = 3)
    return img, height, width, channels

def object_detection(img, net, outputLayers):
    # detects potential objects in image
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    # input detected potential objects in neural network
    net.setInput(blob)
    # get output values from network
    outputs = net.forward(outputLayers)
    # return output ids of objects 
    return outputs

def bounding_boxes(outputs, height, width, classes, confidence):
    boxes, confs, class_ids = [], [], []
    # colors for boxes in BGR
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    
    # finding boxes(box dimensions), confidence levels, class_ids(object ids)
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > confidence:
                cx = int(detect[0] * width)
                cy = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(cx - w/2)
                y = int(cy - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)

    # perform non max suppression on boxes and confidence levels
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
            
    return indexes, boxes, confs, class_ids

def draw_results(img, indexes, boxes, confs, class_ids, classes, width, height):
    # creating boxes
    for i in range(len(boxes)):
        if i in indexes:
            # finds label for object from classes using id
            label = str(classes[class_ids[i]])
            # select box color
            color = (0, 255, 0)
            # dimensions of object box
            x, y, w, h = boxes[i]
            # create box
            cv2.rectangle(img, (x,y), (x+w, y+h), color, math.ceil(min(width, height) * THICKNESS_SCALE))
            # label object box
            cv2.putText(
                img,
                f'{label}: {confs[i]:.2f}',
                (x, y - int(height * TEXT_Y_OFFSET_SCALE)),
                fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                fontScale=min(width, height) * FONT_SCALE,
                thickness=math.ceil(min(width, height) * THICKNESS_SCALE),
                color=color,
            )

    return img

def results_to_json(indexes, boxes, confs, class_ids, classes, img_size):
    return [
            {
                "class_id": int(class_ids[i]),
                "class_name": str(classes[class_ids[i]]),
                "bbox_xywh": boxes[i],  # xywh
                "confidence": confs[i],
                "img_size_wh": img_size,
            }
        for i in range(len(boxes)) if i in indexes
    ]

def image_detection(img_bytes, confidence, draw=False):
    model, classes, output_layers = model_loading() # load yolo model
    image, height, width, _ = image_proccesing(img_bytes) # read and process image
    outputs = object_detection(image, model, output_layers) # detect objects in image
    results = bounding_boxes(outputs, height, width, classes, confidence) # create bounding box and label objects

    if draw:
        labelled_img = draw_results(image, *results, classes, width, height)
        return labelled_img
    
    results = results_to_json(*results, classes, (width, height))
    return results

def encode_img_to_base64(img):
    _, buffer = cv2.imencode('.jpeg', img)
    img_as_text = base64.b64encode(buffer)
    return img_as_text