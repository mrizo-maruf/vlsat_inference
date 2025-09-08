import numpy as np
import open3d as o3d
import json
import matplotlib.pyplot as plt

def generate_cube_surface_points(center, size, n_points):
    """
    Generates a uniformly distributed point cloud on the surface of a cube.

    Args:
        center (np.ndarray): The (x, y, z) center of the cube.
        size (float): The side length of the cube.
        n_points (int): The total number of points to generate.

    Returns:
        np.ndarray: A numpy array of shape (n_points, 3) representing the point cloud.
    """
    points_per_face = n_points // 6
    faces = []
    
    # Generate points for each of the 6 faces
    for axis in range(3):
        for sign in [-1, 1]:
            # Create a 2D grid of points on a face
            p = np.random.rand(points_per_face, 3)
            p[:, axis] = 0.5 * sign # Set one coordinate to the face's plane
            
            # Center the points on the face
            p -= 0.5
            
            # Scale and translate the face to its correct position
            face_points = p * size + center
            faces.append(face_points)
            
    return np.vstack(faces)

def generate_pyramid_surface_points(base_center, base_size, height, n_points):
    """
    Generates a uniformly distributed point cloud on the surface of a square pyramid.

    Args:
        base_center (np.ndarray): The (x, y, z) center of the pyramid's base.
        base_size (float): The side length of the square base.
        height (float): The height of the pyramid.
        n_points (int): The total number of points to generate.

    Returns:
        np.ndarray: A numpy array of shape (n_points, 3) representing the point cloud.
    """
    points_per_surface = n_points // 5
    surfaces = []
    
    # 1. Generate points for the square base
    base_points = np.random.rand(points_per_surface, 3)
    base_points[:, 2] = 0 # Flatten to the base plane (z=0 initially)
    base_points -= 0.5
    base_points *= np.array([base_size, base_size, 0])
    base_points += base_center
    surfaces.append(base_points)
    
    # 2. Define vertices
    apex = base_center + np.array([0, 0, height])
    half = base_size / 2
    v1 = base_center + np.array([-half, -half, 0])
    v2 = base_center + np.array([half, -half, 0])
    v3 = base_center + np.array([half, half, 0])
    v4 = base_center + np.array([-half, half, 0])
    
    base_vertices = [v1, v2, v3, v4]
    
    # 3. Generate points for the 4 triangular faces
    for i in range(4):
        v_a = apex
        v_b = base_vertices[i]
        v_c = base_vertices[(i + 1) % 4]
        
        # Use barycentric coordinates to sample uniformly from a triangle
        r1 = np.random.rand(points_per_surface, 1)
        r2 = np.random.rand(points_per_surface, 1)
        
        # Ensure points are within the triangle
        mask = (r1 + r2) > 1
        r1[mask] = 1 - r1[mask]
        r2[mask] = 1 - r2[mask]
        
        face_points = v_a + r1 * (v_b - v_a) + r2 * (v_c - v_a)
        surfaces.append(face_points)
        
    return np.vstack(surfaces)

def visualize_point_clouds(pcd_list, colors=None):
    """
    Optional: Visualizes a list of point clouds using Open3D.
    """
    if colors is None:
        # Generate random colors if none are provided
        colors = [np.random.rand(3) for _ in pcd_list]

    geometries = []
    for i, pcd_np in enumerate(pcd_list):
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd_np)
        pcd_o3d.paint_uniform_color(colors[i])
        geometries.append(pcd_o3d)
        
    print("Visualizing point clouds. Close the window to continue.")
    o3d.visualization.draw_geometries(geometries)

def save_visualization_matplotlib(pcd_list, output_path, colors=None):
    """
    Saves a 3D scatter plot of the point clouds using Matplotlib.
    """
    if colors is None:
        colors = ['r', 'b', 'g', 'y', 'c', 'm']

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    for i, pcd_np in enumerate(pcd_list):
        x = pcd_np[:, 0]
        y = pcd_np[:, 1]
        z = pcd_np[:, 2]
        ax.scatter(x, y, z, s=1, c=colors[i % len(colors)], label=f'Object {i}')

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.legend()
    ax.set_title('3D Point Cloud Visualization')
    
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Saved Matplotlib visualization to {output_path}")


if __name__ == "__main__":
    # --- 1. Define the Shapes and Their Properties ---
    NUM_POINTS_PER_SHAPE = 5000
    
    # Define a cube
    cube_center = np.array([0, 0, 0.5])
    cube_size = 1.0
    
    # Define a pyramid that will sit on top of the cube
    pyramid_base_size = 0.8
    pyramid_height = 0.7
    # Position the pyramid's base on top of the cube's center
    pyramid_base_center = cube_center + np.array([0, 0, cube_size / 2])

    # --- 2. Generate the Point Clouds ---
    print("Generating point cloud for a cube...")
    cube_pcd = generate_cube_surface_points(cube_center, cube_size, NUM_POINTS_PER_SHAPE)
    
    print("Generating point cloud for a pyramid...")
    pyramid_pcd = generate_pyramid_surface_points(pyramid_base_center, pyramid_base_size, pyramid_height, NUM_POINTS_PER_SHAPE)
    
    # --- 3. Prepare the Input for VL-SAT ---
    # The VL-SAT predictor expects a list of numpy arrays.
    input_for_vlsat = [cube_pcd, pyramid_pcd]
    print(f"\nData prepared for VL-SAT: A list containing {len(input_for_vlsat)} point clouds.")
    print(f" - Object 0 (Cube) shape: {input_for_vlsat[0].shape}")
    print(f" - Object 1 (Pyramid) shape: {input_for_vlsat[1].shape}")

    # --- 4. (Optional) Save to a JSON file for inspection ---
    # This mimics how you might load data for a test
    test_data = {
        "objects": [
            {"id": 0, "name": "cube", "point_cloud": cube_pcd.tolist()},
            {"id": 1, "name": "pyramid", "point_cloud": pyramid_pcd.tolist()}
        ]
    }
    with open("test_shapes.json", "w") as f:
        json.dump(test_data, f)
    print("\nSaved test data to test_shapes.json")

    # --- 5. Save Matplotlib Visualization ---
    save_visualization_matplotlib(input_for_vlsat, "test_shapes_visualization.png", colors=['r', 'b'])


    # --- 6. (Optional) Visualize the Generated Scene with Open3D ---
    visualize_point_clouds(input_for_vlsat, colors=[[1, 0, 0], [0, 0, 1]]) # Red cube, Blue pyramid

    # --- 7. Example of How to Use This with Your Predictor ---
    #
    # # In your build_graph.py, you would do something like this:
    #
    # # a. Load your predictor
    # vlsat_predictor = VLSAT_Predictor(model_path="...", config_path="...", rel_list_path="...")
    #
    # # b. Create mock BBQ objects from the generated point clouds
    # mock_bbq_objects = [
    #     {'id': 0, 'pcd_np': cube_pcd},
    #     {'id': 1, 'pcd_np': pyramid_pcd}
    # ]
    #
    # # c. Call the predict function
    # predicted_edges = vlsat_predictor.predict(mock_bbq_objects)
    #
    # print("\n--- VL-SAT Prediction Results ---")
    # print(json.dumps(predicted_edges, indent=2))
    #
