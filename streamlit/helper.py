from ultralytics import YOLO
import cv2
import argparse
import numpy as np
import math
import cvzone
import json
import pandas as pd

from byte_tracker import BYTETracker
import settings

import streamlit as st
from pytube import YouTube
import tempfile



def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model

def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument('--min_box_area', type=float, default=1.0, help='filter out tiny boxes')
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=10.0,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    
    return parser

def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    return is_display_tracker

def draw_count_zone(image, lanes_coordinates):
    for lane_label, coordinates in lanes_coordinates.items():
        if 'in' in coordinates and 'out' in coordinates:
            cv2.polylines(image, [coordinates['in']], isClosed=True, color=(0, 255, 0), thickness=1)
            cv2.polylines(image, [coordinates['out']], isClosed=True, color=(0, 0, 255), thickness=1)
        cv2.putText(image, f'{lane_label}', tuple(coordinates['out'][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

def draw_counter_box(image, zone_counters):
    cvzone.putTextRect(image, f'LANE A Vehicles IN = {len(zone_counters["A"]["in"])} OUT = {len(zone_counters["A"]["out"])}', [0, 25], thickness=2, scale=1.5, border=2)
    cvzone.putTextRect(image, f'LANE B Vehicles IN = {len(zone_counters["B"]["in"])} OUT = {len(zone_counters["B"]["out"])}', [0, 60], thickness=2, scale=1.5, border=2)
    cvzone.putTextRect(image, f'LANE C Vehicles IN = {len(zone_counters["C"]["in"])} OUT = {len(zone_counters["C"]["out"])}', [0, 95], thickness=2, scale=1.5, border=2)
    cvzone.putTextRect(image, f'LANE D Vehicles IN = {len(zone_counters["D"]["in"])} OUT = {len(zone_counters["D"]["out"])}', [0, 130], thickness=2, scale=1.5, border=2)

def convert_to_np_array(path_str):
    path_list = json.loads(path_str.replace("'", '"'))
    points = [(int(point[1]), int(point[2])) for point in path_list if len(point) == 3]
    return np.array(points)

def process_coord_dataframe(df):
    # if type(df) is not pd.Series:
    #     df = pd.read_csv(df)
    #     df.drop(columns=df.columns[0], axis=1, inplace=True)
    # else:
    df = pd.DataFrame(df)
    df['path'] = df['path'].apply(convert_to_np_array)
    
    lanes = ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D']
    directions = ['in', 'out', 'in', 'out', 'in', 'out', 'in', 'out']

    df['lane'] = lanes
    df['direction'] = directions
    
    lanes_coordinates = {}
    for index, row in df.iterrows():
        lane = row['lane']
        direction = row['direction']
        if lane not in lanes_coordinates:
            lanes_coordinates[lane] = {}
        lanes_coordinates[lane][direction] = row['path']
    
    return lanes_coordinates
    

def _display_detected_frames(conf, model, st_frame, image, model_type, coordinates):
    classnames = ['bus', 'car', 'truck']
    width, height = 1366, 768
        
    lanes_coordinates = coordinates
    
    # track_history = {}
    
    zone_counters = {
        'A': {'in': [], 'out': []},
        'B': {'in': [], 'out': []},
        'C': {'in': [], 'out': []},
        'D': {'in': [], 'out': []},
    }
    
    if 'tracker' not in st.session_state:
        args = make_parser().parse_args()
        st.session_state.tracker = BYTETracker(args, frame_rate=30)
    if 'track_history' not in st.session_state:
        st.session_state.track_history = {}
    tracker = st.session_state.tracker
    track_history = st.session_state.track_history

    image = cv2.resize(image, (width, height))
    detections = np.empty([0, 6])
    
    results = model.predict(source=image, stream=True, conf=conf)

    if model_type == 'Tracking':
        draw_counter_box(image, zone_counters)
    
        draw_count_zone(image, lanes_coordinates)

        for info in results:
            parameters = info.boxes
            for box in parameters:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
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

                cv2.putText(image, f'{class_detect_name}/{confidence}', [x1, y1 - 5], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 1)
                
        track_results = tracker.update(detections, img_info=image.shape, img_size=image.shape)
        
        for track in track_results:
            track_id = track.track_id
            bbox = track.tlbr
            startX, startY, endX, endY = bbox
            startX, startY, endX, endY = int(startX), int(startY), int(endX), int(endY)
            w, h = endX - startX, endY - startY
            cx, cy = startX + w // 2, startY + h // 2
            
            for lane in lanes_coordinates:
                zone_in = lanes_coordinates[lane]['in']
                zone_out = lanes_coordinates[lane]['out']

                if cv2.pointPolygonTest(zone_in, (cx, cy), measureDist=False) == 1:
                    if track_id not in zone_counters[lane]['in']:
                        zone_counters[lane]['in'].append(track_id)

                if cv2.pointPolygonTest(zone_out, (cx, cy), measureDist=False) == 1:
                    if track_id not in zone_counters[lane]['out']:
                        zone_counters[lane]['out'].append(track_id)
            
            if track_id not in track_history:
                track_history[track_id] = []

            track_history[track_id].append((cx, cy))
            if len(track_history[track_id]) > 15:
                track_history[track_id].pop(0)

            for i in range(1, len(track_history[track_id])):
                cv2.line(image, track_history[track_id][i - 1], track_history[track_id][i], (0, 255, 0), 2)

            cv2.putText(image, f'{track_id}', (endX - 15, endY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.circle(image, (cx, cy), 4, (0, 0, 255), -1)
            
        cvzone.putTextRect(image, f'LANE A Vehicles IN = {len(zone_counters["A"]["in"])} OUT = {len(zone_counters["A"]["out"])}', [0, 25], thickness=2, scale=1.5, border=2)
        cvzone.putTextRect(image, f'LANE B Vehicles IN = {len(zone_counters["B"]["in"])} OUT = {len(zone_counters["B"]["out"])}', [0, 60], thickness=2, scale=1.5, border=2)
        cvzone.putTextRect(image, f'LANE C Vehicles IN = {len(zone_counters["C"]["in"])} OUT = {len(zone_counters["C"]["out"])}', [0, 95], thickness=2, scale=1.5, border=2)
        cvzone.putTextRect(image, f'LANE D Vehicles IN = {len(zone_counters["D"]["in"])} OUT = {len(zone_counters["D"]["out"])}', [0, 130], thickness=2, scale=1.5, border=2)
        
    elif model_type == 'Detection':
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

                cv2.putText(image, f'{class_detect_name}/{confidence}', [x1, y1 - 5], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 1)
                cv2.circle(image, (cx, cy), 4, (0, 0, 255), -1)

    st_frame.image(image,
                caption='Detected Video',
                channels="BGR",
                use_column_width=True
                )


def play_webcam(conf, model, model_type):

    source_webcam = settings.WEBCAM_PATH
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            st.session_state.reset_tracker = True
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if not success:
                    vid_cap.release()
                    break
                _display_detected_frames(conf,
                                        model,
                                        st_frame,
                                        image,
                                        model_type,
                                        )
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def play_selected_video(conf, model, model_type):
        
    source_vid = st.sidebar.file_uploader(
        label="Choose a video..."
    )
    
    # upload_coord = st.sidebar.checkbox("Upload Zone Coordinates")
    
    # if not upload_coord:
    #     try:
    #         coordinates = st.session_state['json']
    #     except Exception as e:
    #         st.write()
    # else:
    #     coordinates = st.sidebar.file_uploader(
    #         label = "Upload counting zone coordinates"
    #     )
    
    coordinates = None
    coordinates_file = st.sidebar.file_uploader(
        label = "Upload counting zone coordinates"
    )
        
    if coordinates_file is not None: 
        coordinates = pd.read_csv(coordinates_file)
        coordinates.drop(columns=coordinates.columns[0], axis=1, inplace=True)
        coordinates = process_coord_dataframe(coordinates)
    else:
        coordinates = st.session_state['json']
        coordinates = process_coord_dataframe(coordinates)
    
    if source_vid:
        st.video(source_vid)

    if st.sidebar.button('Detect Video Objects'):
        try:
            tfile = tempfile.NamedTemporaryFile()
            tfile.write(source_vid.read())
            vid_cap = cv2.VideoCapture(tfile.name)
            st.session_state.reset_tracker = True
            st.session_state.reset_track_history = True
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if not success:
                    vid_cap.release()
                    break
                _display_detected_frames(conf,
                                        model,
                                        st_frame,
                                        image,
                                        model_type,
                                        coordinates
                                        )
            
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
