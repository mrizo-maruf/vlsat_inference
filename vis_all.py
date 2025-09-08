import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import argparse
import os
import glob
import re

def visualize_open3d(pcd_list, frame_id, colors=None):
    """
    Displays a list of point clouds in a single interactive Open3D window.

    Args:
        pcd_list (list): A list of numpy arrays, each of shape (N, 3).
        frame_id (str): The ID of the current frame, for the window title.
        colors (list, optional): A list of RGB colors for each point cloud.
    """
    print(f"Displaying frame {frame_id} with Open3D. Press 'Q' in the window to view the next frame.")
    
    if colors is None:
        # Generate random colors if none are provided
        colors = [np.random.rand(3) for _ in pcd_list]

    geometries = []
    for i, pcd_np in enumerate(pcd_list):
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd_np)
        pcd_o3d.paint_uniform_color(colors[i % len(colors)])
        geometries.append(pcd_o3d)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Frame ID: {frame_id}")
    for geom in geometries:
        vis.add_geometry(geom)
    vis.run()
    vis.destroy_window()


def save_matplotlib_3d(pcd_list, output_path, frame_id, colors=None):
    """
    Saves a 3D scatter plot of a list of point clouds to a file using Matplotlib.

    Args:
        pcd_list (list): A list of numpy arrays, each of shape (N, 3).
        output_path (str): The path to save the output image file.
        frame_id (str): The ID of the current frame, for the title.
        colors (list, optional): A list of color strings for each point cloud.
    """
    if colors is None:
        colors = ['b', 'r', 'g', 'y', 'c', 'm']

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    for i, pcd_np in enumerate(pcd_list):
        # Extract object ID from filename for a more informative label
        x = pcd_np[:, 0]
        y = pcd_np[:, 1]
        z = pcd_np[:, 2]
        ax.scatter(x, y, z, s=1, c=colors[i % len(colors)], label=f'Object {i}')

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.legend()
    ax.set_title(f'Multi-Object Visualization - Frame {frame_id}')
    
    try:
        plt.savefig(output_path)
        plt.close(fig)
        print(f"Successfully saved Matplotlib visualization to {output_path}")
    except Exception as e:
        print(f"Error saving Matplotlib visualization: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize a sequence of 3D point cloud frames from a directory."
    )
    parser.add_argument(
        "--input_dir", 
        type=str,
        default="point_clouds",
        help="Path to the directory containing the .npy point cloud files."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="visualization_output",
        help="Directory to save the Matplotlib visualization images."
    )
    
    args = parser.parse_args()

    # --- Find all files and extract unique frame IDs ---
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found at {args.input_dir}")
        exit()
        
    all_files = glob.glob(os.path.join(args.input_dir, "*.npy"))
    if not all_files:
        print(f"Error: No .npy files found in '{args.input_dir}'.")
        exit()

    # Use regex to find timestamps like '_001512_' in filenames
    frame_ids = sorted(list(set(re.findall(r'_(\d{6})_', " ".join(all_files)))))

    if not frame_ids:
        print("Error: Could not parse any frame IDs (e.g., '_001512_') from filenames.")
        exit()

    print(f"Found {len(frame_ids)} unique frames. Processing sequentially...")
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Loop through each frame and visualize ---
    for frame_id in frame_ids:
        search_pattern = os.path.join(args.input_dir, f"*_{frame_id}_*.npy")
        matching_files = sorted(glob.glob(search_pattern))

        print(f"\n--- Processing Frame: {frame_id} ({len(matching_files)} objects) ---")

        all_point_clouds = []
        for file_path in matching_files:
            try:
                loaded_data = np.load(file_path, allow_pickle=True)
                
                point_cloud = None
                if isinstance(loaded_data, np.ndarray) and loaded_data.ndim == 0:
                    loaded_data = loaded_data.item()

                if isinstance(loaded_data, dict):
                    for key in ['pcd', 'points', 'point_cloud', 'pcd_np']:
                        if key in loaded_data and isinstance(loaded_data[key], np.ndarray):
                            point_cloud = loaded_data[key]
                            break
                elif isinstance(loaded_data, np.ndarray):
                    point_cloud = loaded_data
                
                if isinstance(point_cloud, np.ndarray) and point_cloud.ndim == 2 and point_cloud.shape[1] == 3:
                    all_point_clouds.append(point_cloud)
                else:
                    print(f"Warning: Could not extract a valid (N, 3) point cloud from {os.path.basename(file_path)}. Skipping.")

            except Exception as e:
                print(f"Warning: Failed to load or process {os.path.basename(file_path)}. Error: {e}. Skipping.")

        if not all_point_clouds:
            print(f"Warning: No valid point clouds loaded for frame {frame_id}. Skipping.")
            continue

        # --- Visualize the data for the current frame ---
        output_image_path = os.path.join(args.output_dir, f"frame_{frame_id}.png")
        save_matplotlib_3d(all_point_clouds, output_image_path, frame_id)
        visualize_open3d(all_point_clouds, frame_id)

