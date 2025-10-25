import cv2
cv2_version = cv2.__version__
if cv2_version != '4.6.0':
    raise ImportError(f"Expected OpenCV version 4.6.0, but found {cv2_version}. Please install the correct version.")

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg')  # Use a non-interactive backend for environments without display
class ArUcoStructure:
    def __init__(self,name,ids,locations,normals,size):
        self.name = name
        self.ids = ids
        self.locations = locations
        self.normals = normals
        self.size = size
        self.corners, self.axis_x, self.axis_y = self.calculate_aruco_corners()

    def calculate_aruco_corners(self):
        N = self.locations.shape[0]
        corners = np.zeros((N, 4, 3))  # N markers, 4 corners each, 3 coordinates
        axis_x = np.zeros([N, 3])
        axis_y = np.zeros([N, 3])

        half_size = self.size / 2

        for i in range(N):
            # Get the marker center and normal vector
            center = self.locations[i]
            normal = self.normals[i]

            # Normalize the normal vector
            normal = normal / np.linalg.norm(normal)
            
            # Create a coordinate system for the marker
            # First basis vector (pointing y-axis in the marker's local frame)
            if np.allclose(normal, [0, 0, 1]) or np.allclose(normal, [0, 0, -1]):
                # If normal is along z-axis, use x-axis as reference
                v_y = np.array([1, 0, 0])
            else:
                # Otherwise cross with z-axis to get a vector in marker plane
                v_y = np.cross(normal, [0, 0, 1])
                
            if normal[0] > 0:
                v_y = -v_y
            v_y = v_y / np.linalg.norm(v_y)
            axis_y[i] = v_y

            # Second basis vector (pointing x-axis in the marker's local frame)
            v_x = np.cross(normal, v_y)
            v_x = -v_x / np.linalg.norm(v_x)
            axis_x[i] = v_x

            # Calculate the four corners
            # Top-left: center - half_size*v1 - half_size*v2
            corners[i, 0] = center - half_size * v_x + half_size * v_y
            
            # Top-right: center + half_size*v1 - half_size*v2
            corners[i, 1] = center + half_size * v_x + half_size * v_y
            
            # Bottom-right: center + half_size*v1 + half_size*v2
            corners[i, 2] = center + half_size * v_x - half_size * v_y
            
            # Bottom-left: center - half_size*v1 + half_size*v2
            corners[i, 3] = center - half_size * v_x - half_size * v_y

        return corners, axis_x, axis_y

    def calculate_object_points(self,detected_ids):
        objp_list = []
        for id in detected_ids:
            if id in self.ids:
                index = np.where(self.ids == id)[0][0]
                corners = self.corners[index]
                objp_list.append(corners)
        return np.array(objp_list)
    
    def save_structure(self, filename):
        np.savez(filename, ids=self.ids, locations=self.locations, normals=self.normals, size=self.size, name=self.name)

    @staticmethod
    def load_structure(filename):
        data = np.load(filename)
        return ArUcoStructure(data['name'], data['ids'], data['locations'], data['normals'], data['size'])

    @staticmethod
    def generate_ArUco_Ring_Polygon(name, startID, number_of_markers, marker_size, radius):
        # generate a N side polygon ring of markers in the XZ plane center is (0,0,0) and all normal vectors point outwards from the (0,0,0) in XZ plane
        ids = np.arange(startID, startID + number_of_markers)
        angle_increment = 2 * np.pi / number_of_markers
        locations = []
        for i in range(number_of_markers):
            angle = i * angle_increment
            x = radius * np.cos(angle)
            z = radius * np.sin(angle)
            locations.append([x, 0, z])
        locations = np.array(locations)
        normals = []
        for i in range(number_of_markers):
            angle = i * angle_increment
            nx = np.cos(angle)
            nz = np.sin(angle)
            normals.append([nx, 0, nz])
        normals = np.array(normals)
        return ArUcoStructure(name, ids, locations, normals, marker_size)

    @staticmethod
    def generate_Aruco_Grid(name,startID, rows, cols, marker_size, spacing):
        ids = np.arange(startID, startID + rows * cols)
        locations = []
        normals = []
        for r in range(rows):
            for c in range(cols):
                x = c * (marker_size + spacing)
                y = r * (marker_size + spacing)
                z = 0
                locations.append([x, y, z])
                normals.append([0, 0, 1])  # All normals point up in Z direction
        locations = np.array(locations)
        normals = np.array(normals)
        return ArUcoStructure(name,ids, locations, normals, marker_size)

    @staticmethod
    def visualize_aruco_markers(structure):

        # Create 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Define colors for each marker (up to 10 different colors)
        colors = plt.cm.tab10.colors

        # Corner labels
        corner_names = ['TL', 'TR', 'BR', 'BL']

        # Loop through each marker
        for i in range(structure.corners.shape[0]):
            color = colors[i % len(colors)]
            marker_corners = structure.corners[i]
            
            # Plot marker center
            ax.scatter(structure.locations[i, 0], structure.locations[i, 1], structure.locations[i, 2], 
                        color=color, s=100, label=f'Marker {i+1} Center')
            ax.text(structure.locations[i, 0], structure.locations[i, 1], structure.locations[i, 2], 
                    f'Marker {structure.ids[i]}', fontsize=8, color=color)
            
            # Plot normal vector
            normal = structure.normals[i] / np.linalg.norm(structure.normals[i])
            normal_scale = structure.size * 2  # Scale normal vector for visibility
            ax.quiver(structure.locations[i, 0], structure.locations[i, 1], structure.locations[i, 2],
                        normal[0], normal[1], normal[2], 
                        length=normal_scale, color=color, arrow_length_ratio=0.2)
            
            # Plot corners
            ax.scatter(marker_corners[:, 0], marker_corners[:, 1], marker_corners[:, 2], 
                        color=color, marker='o', s=50)
            
            # Label corners
            for j, corner in enumerate(marker_corners):
                ax.text(corner[0], corner[1], corner[2], 
                        f'{structure.ids[i]}-{j}', 
                        fontsize=8, color=color)
            
            # Connect corners to form the marker outline (closed polygon)
            # Extract coordinates for polygon edges
            x = np.append(marker_corners[:, 0], marker_corners[0, 0])
            y = np.append(marker_corners[:, 1], marker_corners[0, 1])
            z = np.append(marker_corners[:, 2], marker_corners[0, 2])
            
            # Plot the marker outline
            ax.plot(x, y, z, '-', color=color, linewidth=2)
            
            # Connect center to corners with dashed lines
            for corner in marker_corners:
                ax.plot([structure.locations[i, 0], corner[0]],
                        [structure.locations[i, 1], corner[1]],
                        [structure.locations[i, 2], corner[2]],
                        '--', color=color, linewidth=1, alpha=0.5)

            # Plot axis x
            ax.plot([structure.locations[i, 0], structure.locations[i, 0] + structure.axis_x[i, 0]],
                    [structure.locations[i, 1], structure.locations[i, 1] + structure.axis_x[i, 1]],
                    [structure.locations[i, 2], structure.locations[i, 2] + structure.axis_x[i, 2]],
                    '-', color='red', linewidth=2)
            # Plot axis y
            ax.plot([structure.locations[i, 0], structure.locations[i, 0] + structure.axis_y[i, 0]],
                    [structure.locations[i, 1], structure.locations[i, 1] + structure.axis_y[i, 1]],
                    [structure.locations[i, 2], structure.locations[i, 2] + structure.axis_y[i, 2]],
                    '-', color='blue', linewidth=2)
        # Set labels and title
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (meters)')
        ax.set_title('ArUco Markers with Corners')

        # Add legend
        ax.legend()

        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])

        # Set reasonable limits based on data
        all_points = np.vstack([structure.locations, structure.corners.reshape(-1, 3)])
        min_vals = all_points.min(axis=0)
        max_vals = all_points.max(axis=0)
        center = (min_vals + max_vals) / 2
        max_range = (max_vals - min_vals).max() / 2

        for axis, lim_center in zip(['x', 'y', 'z'], center):
            getattr(ax, f'set_{axis}lim')(lim_center - max_range * 1.2, lim_center + max_range * 1.2)

        plt.tight_layout()
        return fig, ax

    @staticmethod
    def get_object_points_from_structure(structure, detected_ids):
        objp_list = []
        for i in range(structure.corners.shape[0]):
            objp_list.append(structure.corners[i])
        return np.array(objp_list)

class ArUcoPoseEstimator:
    def __init__(self, camera_matrix, dist_coeffs):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.structureDict = {}
        # OpenCV 4.6 ArUco API
        self.aruco_param = cv2.aruco.DetectorParameters_create()
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)

    def add_structure(self,structure):
        if structure.name not in self.structureDict:
            self.structureDict[structure.name] = structure
    
    def estimate_pose(self, image):
        imgp, ids, rejected = cv2.aruco.detectMarkers(image, self.aruco_dict, parameters=self.aruco_param)
        results = {}
        for structure in self.structureDict.values():
            if ids is None:
                results[structure.name] = None
                continue
            
            for id in ids[:, 0]:
                if id not in structure.ids:
                    index = np.where(ids[:, 0] == id)
                    ids = np.delete(ids, index, axis=0)
                    imgp = np.delete(imgp, index, axis=0)

            if len(ids) == 0:
                results[structure.name] = None
                continue
        
            imgp = np.array(imgp)
            imgp = imgp.reshape(-1, 2)
            objp = structure.calculate_object_points(ids)

            _, rvec, tvec = cv2.solvePnP(objp, imgp, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

            results[structure.name] = {
                "p": tvec,
                "r": rvec
            }

        return results

    def draw_detected_markers(self, image):
        corners, ids, rejected = cv2.aruco.detectMarkers(
            image,
            self.aruco_dict,
            parameters=self.aruco_param,
        )
        frame_data = {}
        
        for i in range(len(corners)):
            marker_info = (f"ID: {ids[i]}")
            frame_data[ids[i][0]] = corners[i]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2
            color = (0, 0, 255)
            cv2.putText(image, marker_info, (int(corners[i][0][0][0]), int(corners[i][0][0][1])), font, font_scale, color, thickness, cv2.LINE_AA)
        return image

    def draw_estimated_poses(self, image, results):
        for structure in self.structureDict.values():
            if structure.name in results and results[structure.name] is not None:
                rvec = results[structure.name]["r"]
                tvec = results[structure.name]["p"]
                cv2.drawFrameAxes(image, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.1)

            structure_info = (f"Name: {structure.name}")
            imgp = cv2.projectPoints(structure.corners, rvec, tvec, self.camera_matrix, self.dist_coeffs)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2
            color = (0, 0, 255)
            cv2.putText(image, structure_info, (int(imgp[0][0][0]), int(imgp[0][0][1])), font, font_scale, color, thickness, cv2.LINE_AA)
        return image

if __name__ == "__main__":
    # Example usage
    marker_size = 0.056
    radius = 0.11
    # fig, ax = ArUcoStructure.visualize_aruco_markers(structure)
    # plt.show()

    camera_matrix = pickle.load(open("real_time_localization/parameters/mtx_cam1.pkl", "rb"))
    dist_coeffs = pickle.load(open("real_time_localization/parameters/dist_cam1.pkl", "rb"))

    estimator = ArUcoPoseEstimator(camera_matrix, dist_coeffs)
    structure = ArUcoStructure.generate_ArUco_Ring_Polygon("Ring1", 16, 6, marker_size, radius)
    estimator.add_structure(structure)

    camera = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                break
            # results = estimator.estimate_pose(frame)
            frame = estimator.draw_detected_markers(frame)
            # frame = estimator.draw_estimated_poses(frame, results)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(e)
    finally:
        camera.release()
        cv2.destroyAllWindows()
