import cv2
from gui_buttons import Buttons

# Initialize Buttons
button = Buttons()
button.add_button("bottle", 20,100)
button.add_button("cup", 20, 20)
button.add_button("bird", 20, 180)
button.add_button("cat", 20, 260)
button.add_button("horse", 20, 340)
button.add_button("dog", 20, 420)
button.add_button("scissors", 1000, 20)
button.add_button("keyboard", 950, 100)
button.add_button("phone", 1000, 180)
button.add_button("bus", 1000, 260)







colors = button.colors


# Opencv DNN
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)

# Load class lists
classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

print("Objects list")
print(classes)


# Initialize camera
cap = cv2.VideoCapture('rtsp://admin:19981105@192.168.1.108', cv2.CAP_DSHOW)#( 'G:\course_320\视频素材参考\CF.mp4') #   2, cv2.CAP_DSHOW
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,650)
# FULL HD 1920 x 1080


def click_button(event, x, y, flags, params):
    global button_person
    if event == cv2.EVENT_LBUTTONDOWN:
        button.button_click(x, y)

# Create window
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", click_button)

while True:
    # Get frames
    ret, frame = cap.read()

    # Get active buttons list
    active_buttons = button.active_buttons_list()
    #print("Active buttons", active_buttons)

    # Object Detection
    (class_ids, scores, bboxes) = model.detect(frame, confThreshold=0.3, nmsThreshold=.4)
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        (x, y, w, h) = bbox
        class_name = classes[class_id]
        color = colors[class_id]

        if class_name in active_buttons:
            cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 5)


    # Display buttons
    button.display_buttons(frame)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 30:
        break

cap.release()
cv2.destroyAllWindows()
