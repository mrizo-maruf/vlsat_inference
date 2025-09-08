import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import argparse
import os
import glob

def visualize_open3d(pcd_list, colors=None):
    """
    Displays a list of point clouds in a single interactive Open3D window.

    Args:
        pcd_list (list): A list of numpy arrays, each of shape (N, 3).
        colors (list, optional): A list of RGB colors for each point cloud.
    """
    print("Displaying point clouds with Open3D. Press 'Q' in the window to close.")
    
    if colors is None:
        # Generate random colors if none are provided
        colors = [np.random.rand(3) for _ in pcd_list]

    geometries = []
    for i, pcd_np in enumerate(pcd_list):
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd_np)
        pcd_o3d.paint_uniform_color(colors[i % len(colors)])
        geometries.append(pcd_o3d)
    
    o3d.visualization.draw_geometries(geometries)

def save_matplotlib_3d(pcd_list, output_path, colors=None):
    """
    Saves a 3D scatter plot of a list of point clouds to a file using Matplotlib.

    Args:
        pcd_list (list): A list of numpy arrays, each of shape (N, 3).
        output_path (str): The path to save the output image file.
        colors (list, optional): A list of color strings for each point cloud.
    """
    if colors is None:
        colors = ['b', 'r', 'g', 'y', 'c', 'm']

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    for i, pcd_np in enumerate(pcd_list):
        x = pcd_np[:, 0]
        y = pcd_np[:, 1]
        z = pcd_np[:, 2]
        ax.scatter(x, y, z, s=1, c=colors[i % len(colors)], label=f'Object ID {i}')

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.legend()
    ax.set_title('Multi-Object Point Cloud Visualization')
    
    try:
        plt.savefig(output_path)
        plt.close(fig)
        print(f"Successfully saved Matplotlib visualization to {output_path}")
    except Exception as e:
        print(f"Error saving Matplotlib visualization: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize multiple 3D point clouds from a specific frame."
    )
    parser.add_argument(
        "input_dir", 
        type=str,
        default = "/home/rizo/mipt_ccm/vlsat_inference/point_clouds",
        help="Path to the directory containing the .npy point cloud files."
    )
    parser.add_argument(
        "--frame_id", 
        type=str,
        required=True,
        help="The common identifier in the filenames to visualize (e.g., '001557')."
    )
    parser.add_argument(
        "--output_image", 
        type=str, 
        default="pointcloud_visualization.png",
        help="Path to save the Matplotlib visualization image."
    )
    
    args = parser.parse_args()

    # --- Find all matching files ---
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found at {args.input_dir}")
        exit()
        
    search_pattern = os.path.join(args.input_dir, f"*_{args.frame_id}_*.npy")
    matching_files = glob.glob(search_pattern)

    if not matching_files:
        print(f"Error: No .npy files found in '{args.input_dir}' with frame_id '{args.frame_id}'.")
        exit()

    print(f"Found {len(matching_files)} files for frame '{args.frame_id}'. Loading...")

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
        print("Error: Could not load any valid point clouds. Exiting.")
        exit()

    print(f"Successfully loaded {len(all_point_clouds)} point clouds.")

    # --- Visualize the data ---
    save_matplotlib_3d(all_point_clouds, args.output_image)
    visualize_open3d(all_point_clouds)
