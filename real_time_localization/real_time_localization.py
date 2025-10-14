import cv2
import json
import numpy as np
import os
import pickle
from datetime import datetime
import tqdm


class Capture:
    def __init__(self, cameras = None):
        try:
            with open('setting.json', 'r') as f:
                self.settings = json.load(f)
        except:
            raise Exception(f"Error loading settings from setting.json")
        
        self.cameras = cameras
        if cameras is not None:
            self.reference_shape = self.cameras[0].read()[1].shape[:2]
        else:
            self.reference_shape = None
        

    def __del__(self):
        if self.cameras is not None:
            for i in range(len(self.cameras)):
                self.cameras[i].release()
            
    def __str__(self):
        return json.dumps(self.settings, indent=4)
    
    def open_cameras(self, cameras):
        self.cameras = cameras
        if cameras is not None:
            self.reference_shape = self.cameras[0].read()[1].shape[:2]
        else:
            self.reference_shape = None
            
    def default_capture(self, frame, camera_name):
        return frame
    
    def reproduce_capture(self, capture_function, video_path):
        cap = cv2.VideoCapture(video_path)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.reference_shape = (self.height, self.width//len(self.settings['cameras']))
        out = cv2.VideoWriter(f"output/{video_path.split('/')[-1].split('.')[0]}_reproduce.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 24, (self.width, self.height))
        data = []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for _ in tqdm.tqdm(range(total_frames)):
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_data = {}
            frames = self.frame_slicing(frame)
            for i in range(len(self.settings['cameras'])):
                camera_name = self.settings['cameras'][i]
                frame_data[camera_name] = capture_function(frames[i], camera_name)
            data.append(frame_data)
            
            out.write(self.frame_concatent(frames, self.reference_shape))

        json_data = []
        for frame_data in data:
            frame_json = {}
            for camera_name, positions in frame_data.items():
                frame_json[camera_name] = [pos.tolist() for pos in positions]
            json_data.append(frame_json)

        with open(f"output/{video_path.split('/')[-1].split('.')[0]}_reproduce.json", 'w') as f:
            json.dump(json_data, f, indent=2)
            
        out.release()
        cap.release()
    
    def save_video(self, capture_function = None, save_preview = False):
        if capture_function is None:
            capture_function = self.default_capture
            
        frames = []
            
        for i in range(len(self.cameras)):
            self.cameras[i].set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cameras[i].set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
            if not self.cameras[i].isOpened():
                print(f"Error: Could not open camera {i}")
                return
            ret, frame = self.cameras[i].read()
            frames.append(frame)
        
        frame = self.frame_concatent(frames, self.reference_shape)
        self.height, self.width = frame.shape[:2]
        
        if not os.path.exists('output'):
            os.makedirs('output')
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'output/{timestamp}.mp4'
        
        if len(self.settings['cameras']) == 1:
            filename = f"output/{self.settings['cameras'][0]}_{timestamp}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, 24, (self.width, self.height))

        data = []
        while True:
            frames = []
            show_frames = []
            frame_data = {}
            
            for i in range(len(self.cameras)):
                camera_name = self.settings['cameras'][i]
                ret, frame = self.cameras[i].read()
                frames.append(frame)
                if not save_preview:
                    shown_frame = frames[i].copy()
                else:
                    shown_frame = frame
                frame_data[camera_name] = capture_function(shown_frame, camera_name)
                show_frames.append(shown_frame)
                
            frame = self.frame_concatent(frames, self.reference_shape)
            show_frame = self.frame_concatent(show_frames, self.reference_shape)
            out.write(frame)
            cv2.imshow('Frames', show_frame)
            
            data.append(frame_data)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        json_data = []
        for frame_data in data:
            frame_json = {}
            for camera_name, positions in frame_data.items():
                frame_json[camera_name] = [pos.tolist() for pos in positions]
            json_data.append(frame_json)

        with open(f'output/{timestamp}.json', 'w') as f:
            json.dump(json_data, f, indent=2)
            
        out.release()
        cv2.destroyAllWindows()
        
    def frame_concatent(self, frames, reference_shape):
        for i in range(len(frames)):
            frames[i] = cv2.resize(frames[i], (reference_shape[1], reference_shape[0]))
        return np.concatenate(frames, axis=1)
    
    def frame_slicing(self, frame):
        frames = []
        width = frame.shape[1] // len(self.settings['cameras'])
        height = frame.shape[0]
        for i in range(len(self.settings['cameras'])):
            frames.append(frame[:, i * width:(i + 1) * width])
        return frames
    
    
class Localization(Capture):
    def __init__(self, cameras = None):
        super().__init__(cameras)
        self.cameras_mtx = {}
        self.cameras_dist = {}
        self.cameras_extrinsic = {}
        for camera_name in self.settings['cameras']:
            self.cameras_mtx[camera_name] = pickle.load(open(f'{self.settings["camera_parameter_path"]}/mtx_{camera_name}.pkl', 'rb'))
            self.cameras_dist[camera_name] = pickle.load(open(f'{self.settings["camera_parameter_path"]}/dist_{camera_name}.pkl', 'rb'))
            self.cameras_extrinsic[camera_name] = pickle.load(open(f'{self.settings["camera_parameter_path"]}/extrinsic_{camera_name}.pkl', 'rb'))
        
        self.detect_param_localization = cv2.aruco.DetectorParameters()
        self.detect_dict_localization = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.settings['aruco_dict_localization']))
        self.aruco_detector_localization = cv2.aruco.ArucoDetector(self.detect_dict_localization, self.detect_param_localization)
        
    def localization(self, frame, camera_name):
        corners, ids, rejected = cv2.aruco.detectMarkers(frame, self.detect_dict_localization, parameters=self.detect_param_localization)
        frame_data = {}
        
        for i in range(len(corners)):
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], self.settings['marker_size_localization'], self.cameras_mtx[camera_name], self.cameras_dist[camera_name])
            homogeneous_marker_point = np.eye(4)
            homogeneous_marker_point[:3, :3] = rvec
            homogeneous_marker_point[:3, 3] = tvec
            
            homogeneous_marker_point = self.cameras_extrinsic[camera_name] @ homogeneous_marker_point
            marker_info = (f"ID: {ids[i]} X: {homogeneous_marker_point[:3, 3][0]:.2f} Y: {homogeneous_marker_point[:3, 3][1]:.2f} Z: {homogeneous_marker_point[:3, 3][2]:.2f}")
            frame_data[ids[i][0]] = homogeneous_marker_point[:3, 3]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2
            color = (0, 0, 255)
            cv2.putText(frame, marker_info, (int(corners[i][0][0][0]), int(corners[i][0][0][1])), font, font_scale, color, thickness, cv2.LINE_AA)
            
        return frame_data

    def detection(self, frame, camera_name):
        corners, ids, rejected = cv2.aruco.detectMarkers(frame, self.detect_dict_localization, parameters=self.detect_param_localization)
        frame_data = {}
        
        for i in range(len(corners)):
            marker_info = (f"ID: {ids[i]}")
            frame_data[ids[i][0]] = corners[i]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2
            color = (0, 0, 255)
            cv2.putText(frame, marker_info, (int(corners[i][0][0][0]), int(corners[i][0][0][1])), font, font_scale, color, thickness, cv2.LINE_AA)
        
        return frame_data

    def generate_objp():
        pass
        
def main():
    localization = Localization([cv2.VideoCapture(0)])
    localization.save_video(localization.detection, save_preview=False)
    
    # localization = Localization()
    # localization.reproduce_capture(localization.localization, 'output/20251001_183755.mp4')
if __name__ == "__main__":
    main()
