import matplotlib.pyplot as plt
import numpy as np
import json
import os

class ResultAnalysis:
    def __init__(self, file_path):
        self.data = {}
        
        try:
            with open('setting.json', 'r') as f:
                self.settings = json.load(f)
        except:
            raise Exception(f"Error loading settings from setting.json")
        
        self.load_data(file_path)
        self.remove_results()

    def load_data(self, file_path):
        with open(file_path, 'r') as f:
            self.data = json.load(f)
            
    def remove_results(self):
        os.makedirs('result/', exist_ok=True)
        files = os.listdir('result/')
        for file in files:
            os.remove(f'result/{file}')
            
    def calculate_inter_camera_difference(self, camera_name):
        camera_coordinates = []
        center_camera_coordinates = []
        for frame_data in self.data:
            if len(frame_data[camera_name]) > 0 and len(frame_data[self.settings['center_camera']]) > 0:
                camera_coordinates.append(frame_data[camera_name])
                center_camera_coordinates.append(frame_data[self.settings['center_camera']])
                
        camera_coordinates = np.array(camera_coordinates)[:, 0, :]
        center_camera_coordinates = np.array(center_camera_coordinates)[:, 0, :]
        difference = camera_coordinates - center_camera_coordinates
        difference_mean = np.mean(difference, axis=0)
        difference_std = np.std(difference, axis=0)
        print(f'Difference between {camera_name} and {self.settings["center_camera"]} mean: {difference_mean}')
        print(f'Difference between {camera_name} and {self.settings["center_camera"]} std: {difference_std}')
        os.makedirs('result/', exist_ok=True)
        with open(f'result/difference.log', 'a') as f:
            f.write(f'Difference between {camera_name} and {self.settings["center_camera"]} mean: {difference_mean}\n')
            f.write(f'Difference between {camera_name} and {self.settings["center_camera"]} std: {difference_std}\n')
        
    def visualize_points(self, camera_name):
        camera_coordinates = []
        for frame_data in self.data:
            if len(frame_data[camera_name]) > 0:
                camera_coordinates.append(frame_data[camera_name])
                
        camera_coordinates = np.array(camera_coordinates)[:, 0, :]
        
        fig = plt.figure(figsize=(10, 10))
        self.draw_points(fig, 1, camera_name, camera_coordinates, 0)
        self.draw_points(fig, 2, camera_name, camera_coordinates, 1)
        self.draw_points(fig, 3, camera_name, camera_coordinates, 2)
        plt.savefig(f'result/visualize_points_{camera_name}.png')
        plt.close(fig)
        
    def visualize_points_multi_cameras(self):
        fig = plt.figure(figsize=(10, 10))
        camera_coordinates = {f'{camera_name}': [] for camera_name in self.settings['cameras']}
        for frame_data in self.data:
            for camera_name in self.settings['cameras']:
                if len(frame_data[camera_name]) > 0:
                    camera_coordinates[camera_name].append(frame_data[camera_name])
        
        for camera_name in self.settings['cameras']:
            camera_coordinates[camera_name] = np.array(camera_coordinates[camera_name])[:, 0, :]
        
        self.draw_points_multi_cameras(fig, 1, camera_coordinates, 0)
        self.draw_points_multi_cameras(fig, 2, camera_coordinates, 1)
        self.draw_points_multi_cameras(fig, 3, camera_coordinates, 2)
        plt.savefig(f'result/visualize_points_multi_cameras.png')
        plt.close(fig)
    
    def draw_points(self, fig, i, camera_name, points, dimension_erase):
        dimenstion_kept = [0, 1, 2]
        dimenstion_kept.remove(dimension_erase)
        xyz_array = ['X', 'Y', 'Z']
        
        ax = fig.add_subplot(1, 3, i)
        ax.set_box_aspect(1)
        
        ax.set_xlabel(xyz_array[dimenstion_kept[0]])
        ax.set_ylabel(xyz_array[dimenstion_kept[1]])
        ax.scatter(points[:, dimenstion_kept[0]], points[:, dimenstion_kept[1]], color='red', alpha=0.5)
        ax.set_title(f'{camera_name}')
        
    
    def draw_points_multi_cameras(self, fig, i, points, dimension_erase):
        dimenstion_kept = [0, 1, 2]
        dimenstion_kept.remove(dimension_erase)
        xyz_array = ['X', 'Y', 'Z']
        colors = ['red', 'green', 'blue', 'yellow', 'purple']
        
        ax = fig.add_subplot(1, 3, i)
        ax.set_box_aspect(1)

        ax.set_xlabel(xyz_array[dimenstion_kept[0]])
        ax.set_ylabel(xyz_array[dimenstion_kept[1]])
        for camera_name in self.settings['cameras']:
            ax.scatter(points[camera_name][:, dimenstion_kept[0]], points[camera_name][:, dimenstion_kept[1]], color=colors[self.settings['cameras'].index(camera_name)], alpha=0.5)
        ax.set_title(f'Multi Cameras')
        ax.legend(self.settings['cameras'])
        
def main():
    result_analysis = ResultAnalysis('output/20251001_183755_reproduce.json')
    result_analysis.calculate_inter_camera_difference('cam0')
    result_analysis.calculate_inter_camera_difference('cam1')
    result_analysis.calculate_inter_camera_difference('cam2')
    result_analysis.visualize_points('cam0')
    result_analysis.visualize_points('cam1')
    result_analysis.visualize_points('cam2')
    result_analysis.visualize_points_multi_cameras()
    
if __name__ == "__main__":
    main()