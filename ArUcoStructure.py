import cv2
cv2_version = cv2.__version__
if cv2_version != '4.6.0':
    raise ImportError(f"Expected OpenCV version 4.6.0, but found {cv2_version}. Please install the correct version.")
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
        self.corners = self.calculate_aruco_corners()

    def calculate_aruco_corners(self):
        N = self.locations.shape[0]
        corners = np.zeros((N, 4, 3))  # N markers, 4 corners each, 3 coordinates

        half_size = self.size / 2

        for i in range(N):
            # Get the marker center and normal vector
            center = self.locations[i]
            normal = self.normals[i]

            # Normalize the normal vector
            normal = normal / np.linalg.norm(normal)
            
            # Create a coordinate system for the marker
            # First basis vector (pointing right in the marker's local frame)
            if np.allclose(normal, [0, 0, 1]) or np.allclose(normal, [0, 0, -1]):
                # If normal is along z-axis, use x-axis as reference
                v1 = np.array([1, 0, 0])
            else:
                # Otherwise cross with z-axis to get a vector in marker plane
                v1 = np.cross(normal, [0, 0, 1])
                
            v1 = v1 / np.linalg.norm(v1)
            
            # Second basis vector (pointing down in the marker's local frame)
            v2 = np.cross(normal, v1)
            v2 = v2 / np.linalg.norm(v2)
            
            # Calculate the four corners
            # Top-left: center - half_size*v1 - half_size*v2
            corners[i, 0] = center - half_size * v1 - half_size * v2
            
            # Top-right: center + half_size*v1 - half_size*v2
            corners[i, 1] = center + half_size * v1 - half_size * v2
            
            # Bottom-right: center + half_size*v1 + half_size*v2
            corners[i, 2] = center + half_size * v1 + half_size * v2
            
            # Bottom-left: center - half_size*v1 + half_size*v2
            corners[i, 3] = center - half_size * v1 + half_size * v2

        return corners

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
    def generate_ArUco_Ring_Polygon(startID, number_of_markers, marker_size, spacing):
        # generate a N side polygon ring of markers in the XZ plane center is (0,0,0) and all normal vectors point outwards from the (0,0,0) in XZ plane
        ids = np.arange(startID, startID + number_of_markers)
        angle_increment = 2 * np.pi / number_of_markers
        radius = (marker_size + spacing) / (2 * np.sin(np.pi / number_of_markers))
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
        return ArUcoStructure(ids, locations, normals, marker_size)

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
            
            # Label corners if requested
            # for j, corner in enumerate(marker_corners):
            #     ax.text(corner[0], corner[1], corner[2], 
            #             f'M{i+1}-{corner_names[j]}', 
            #             fontsize=8, color=color)
            
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

class ArUcoPoseEstimator:
    def __init__(self, camera_matrix, dist_coeffs):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters_create()
        self.structureDict = {}

    def add_structure(self,structure):
        if structure.name not in self.structureDict:
            self.structureDict[structure.name] = structure
    
    def estimate_pose(self, image):
        raise NotImplementedError("Pose estimation method not implemented yet.")
        return {
            "name":{"p":np.array([0,0,0]),"r":np.array([0,0,0])}
        }
    def draw_detected_markers(self, image, results):
        raise NotImplementedError("Draw detected markers method not implemented yet.")
        return image

if __name__ == "__main__":
    # Example usage
    marker_size = 0.05  # 5 cm
    spacing = 0.01      # 1 cm
    structure = ArUcoStructure.generate_Aruco_Grid("Grid1", 10, 3, 4, marker_size, spacing)
    fig, ax = ArUcoStructure.visualize_aruco_markers(structure)
    plt.show()