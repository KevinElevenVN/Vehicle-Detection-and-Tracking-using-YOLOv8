import argparse
import cv2
import cvzone
import math
import numpy as np
from ultralytics import YOLO
from byte_tracker import BYTETracker


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=50, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument('--min_box_area', type=float, default=1.0, help='filter out tiny boxes')
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=10.0,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    
    parser.add_argument(
        "--save",
        action="store_true",
        help="whether to save the inference result of image/video",
    )
    
    return parser


video_path = r'D:\Works\Year 3 - Semester 3\Graduation_Thesis\code\video\city_traffic.mp4'
cap = cv2.VideoCapture(video_path)
model = YOLO(r'D:\Works\Year 3 - Semester 3\Graduation_Thesis\code\YOLO_model\trained_yolov8n_S640R15.pt')

classnames = ['bus', 'car', 'truck']

lanes_coordinates = {
    'A': {'in': np.array([(508, 509), (570, 491), (463, 451), (400, 463)]),
        'out': np.array([(304, 442), (260, 496), (313, 540), (471, 512)])},
    'B': {'in': np.array([(230, 588), (15, 605), (17, 628), (245, 612)]),
        'out': np.array([(15, 631), (253, 616), (288, 655), (17, 671)])},
    'C': {'in': np.array([(740, 724), (859, 684), (975, 731), (838, 767)]),
        'out': np.array([(899, 668), (1023, 623), (1122, 655), (1001, 712)])},
    'D': {'in': np.array([(887, 543), (975, 571), (1009, 562), (952, 531)]),
        'out': np.array([(863, 541), (803, 519) , (912, 499), (951, 518)])},
}

args = make_parser().parse_args()

tracker = BYTETracker(args, frame_rate=30)

track_history = {}

zone_counters = {
    'A': {'in': [], 'out': []},
    'B': {'in': [], 'out': []},
    'C': {'in': [], 'out': []},
    'D': {'in': [], 'out': []},
}

width, height = 1366, 768

# save_path = r'D:\Works\Year 3 - Semester 3\Graduation_Thesis\code\output\track_bytetrack.mp4'
# fps = cap.get(cv2.CAP_PROP_FPS)
# fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# vid_writer = cv2.VideoWriter(
#     save_path, fourcc, fps, (int(width), int(height))
# )

if args.save:
    save_path = r'D:\Works\Year 3 - Semester 3\Graduation_Thesis\code\output\track_bytetrack.mp4'
    fps = cap.get(cv2.CAP_PROP_FPS)
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )

'''Time in second'''
time = 0
index = time * cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_POS_FRAMES, index)


while cap.isOpened():    
    
    list_detect = ['car', 'bus', 'truck']
    classes = [classnames.index(i) for i in list_detect]
    conf_threshold = 0.6

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (width, height))

    results = model(source=frame, conf=conf_threshold)
    detections = np.empty([0, 6])

    '''Drawing count zone'''
    for lane_label, coordinates in lanes_coordinates.items():
        if 'in' in coordinates and 'out' in coordinates:
            cv2.polylines(frame, [coordinates['in']], isClosed=True, color=(0, 255, 0), thickness=1)
            cv2.polylines(frame, [coordinates['out']], isClosed=True, color=(0, 0, 255), thickness=1)

        cv2.putText(frame, f'{lane_label}', tuple(coordinates['out'][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            w, h = x2 - x1, y2 - y1
            confidence = math.ceil(box.conf[0] * 100)
            class_detect = int(box.cls[0])
            class_detect_name = classnames[class_detect]
            if class_detect_name == 'car':
                class_detect_name = 'c'
            elif class_detect_name == 'truck':
                class_detect_name = 't'
            else:
                class_detect_name = 'b'
            
            new_detections = np.array([x1, y1, x2, y2, confidence, class_detect])
            detections = np.vstack([detections, new_detections])

            cv2.putText(frame, f'{class_detect_name}/{confidence}', [x1, y1 - 5], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 1)

    # Update ByteTrack with detections
    track_results = tracker.update(detections, img_info=frame.shape, img_size=frame.shape)

    for track in track_results:
        track_id = track.track_id
        bbox = track.tlbr
        startX, startY, endX, endY = bbox
        startX, startY, endX, endY = int(startX), int(startY), int(endX), int(endY)
        w, h = endX - startX, endY - startY
        cx, cy = startX + w // 2, startY + h // 2

        if track_id not in track_history:
            track_history[track_id] = []

        track_history[track_id].append((cx, cy))
        if len(track_history[track_id]) > 10:
            track_history[track_id].pop(0)

        for i in range(1, len(track_history[track_id])):
            cv2.line(frame, track_history[track_id][i - 1], track_history[track_id][i], (0, 255, 0), 2)

        for lane in lanes_coordinates:
            zone_in = lanes_coordinates[lane]['in']
            zone_out = lanes_coordinates[lane]['out']

            if cv2.pointPolygonTest(zone_in, (cx, cy), measureDist=False) == 1:
                if track_id not in zone_counters[lane]['in']:
                    zone_counters[lane]['in'].append(track_id)

            if cv2.pointPolygonTest(zone_out, (cx, cy), measureDist=False) == 1:
                if track_id not in zone_counters[lane]['out']:
                    zone_counters[lane]['out'].append(track_id)

        cv2.putText(frame, f'{track_id}', (endX - 15, endY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    cvzone.putTextRect(frame, f'LANE A Vehicles IN = {len(zone_counters["A"]["in"])} OUT = {len(zone_counters["A"]["out"])}', [0, 25], thickness=2, scale=1.5, border=2)
    cvzone.putTextRect(frame, f'LANE B Vehicles IN = {len(zone_counters["B"]["in"])} OUT = {len(zone_counters["B"]["out"])}', [0, 60], thickness=2, scale=1.5, border=2)
    cvzone.putTextRect(frame, f'LANE C Vehicles IN = {len(zone_counters["C"]["in"])} OUT = {len(zone_counters["C"]["out"])}', [0, 95], thickness=2, scale=1.5, border=2)
    cvzone.putTextRect(frame, f'LANE D Vehicles IN = {len(zone_counters["D"]["in"])} OUT = {len(zone_counters["D"]["out"])}', [0, 130], thickness=2, scale=1.5, border=2)

    if args.save:
        vid_writer.write(frame)

    cv2.imshow('frame', frame)
        
    ch = cv2.waitKey(1)
    if ch == 27 or ch == ord("q") or ch == ord("Q"):
        break


cap.release()
cv2.destroyAllWindows()
