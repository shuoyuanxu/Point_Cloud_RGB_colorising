import yaml
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_matrix(data):
    """Load a 4x4 matrix from nested list format."""
    return np.array(data).reshape((4, 4))

def plot_frame(ax, T, label, color):
    """Plot a coordinate frame given a transformation matrix T (4x4)."""
    origin = T[:3, 3]
    x_axis = T[:3, 0]
    y_axis = T[:3, 1]
    z_axis = T[:3, 2]

    ax.quiver(*origin, *x_axis, length=0.1, color=color, label=f'{label} x')
    ax.quiver(*origin, *y_axis, length=0.1, color='g', label=f'{label} y')
    ax.quiver(*origin, *z_axis, length=0.1, color='b', label=f'{label} z')
    ax.text(*origin, f'{label}', fontsize=10)

def main():
    config_path = "config.yaml"  # Update if needed

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    T_right = load_matrix(cfg['T_lidar_camera_right'])
    T_left  = load_matrix(cfg['T_lidar_camera_left'])
    T_lidar = np.eye(4)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("LiDAR and Camera Frames")

    plot_frame(ax, T_lidar, 'LiDAR', 'r')
    plot_frame(ax, T_right, 'Right Cam', 'orange')
    plot_frame(ax, T_left, 'Left Cam', 'purple')

    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([-0.5, 0.5])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

