import numpy as np
import open3d as o3d
import argparse
import os
import gzip
import pickle

# Helper to create a small 3D text label (best-effort; will silently skip if unsupported)
def _create_text_label(text, position, color=(1,1,1)):
    try:
        # Common font path on many Linux systems; adjust if needed
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        if not os.path.exists(font_path):
            font_path = None  # let Open3D pick a default if possible
        # Open3D 0.17+ provides create_text
        mesh = o3d.geometry.TriangleMesh.create_text(text=text, depth=0.002, font_size=48, font=font_path)
        mesh.compute_vertex_normals()
        # Center the text about origin before scaling / translating
        aabb = mesh.get_axis_aligned_bounding_box()
        mesh.translate(-aabb.get_center())
        # Scale down to a reasonable size
        mesh.scale(0.005, center=(0,0,0))
        mesh.paint_uniform_color(color)
        mesh.translate(position)
        return mesh
    except Exception:
        return None

def visualize_bbq_objects(bbq_objects, show_bbox=True, show_center=False, print_color_mapping=False, print_edges=True):
    """
    Displays a list of BBQ objects in a single interactive Open3D window.
    Each object is given a different color and its bounding box and center are shown.
    Additionally, object IDs are rendered as 3D text above the bounding box when supported.

    Args:
        bbq_objects (list): The list of object dictionaries loaded from the pickle file.
        show_bbox (bool): Whether to show bounding boxes.
        show_center (bool): Whether to show center spheres and labels.
        print_color_mapping (bool): Whether to print object ID to color mapping.
        print_edges (bool): Whether to print edge information.
    """
    if not bbq_objects:
        print("The list of objects is empty. Nothing to visualize.")
        return

    print(f"Visualizing {len(bbq_objects)} objects with Open3D. Press 'Q' in the window to close.")
    
    # Choose color map or fallback to deterministic random colors when matplotlib isn't available
    if 'plt' in globals() and plt is not None:
        color_map = plt.get_cmap("tab10")
        _rand_color = None
    else:
        color_map = None
        import random
        def _rand_color(i):
            rnd = random.Random(i)
            return (rnd.random(), rnd.random(), rnd.random())

    geometries = []
    
    # Define a large set of distinct colors with names
    color_palette = [
        ([1.0, 0.0, 0.0], "red"),
        ([0.0, 1.0, 0.0], "green"), 
        ([0.0, 0.0, 1.0], "blue"),
        ([1.0, 1.0, 0.0], "yellow"),
        ([1.0, 0.0, 1.0], "magenta"),
        ([0.0, 1.0, 1.0], "cyan"),
        ([1.0, 0.5, 0.0], "orange"),
        ([0.5, 0.0, 0.5], "purple"),
        ([1.0, 0.75, 0.8], "pink"),
        ([0.5, 0.5, 0.5], "gray"),
        ([0.6, 0.4, 0.2], "brown"),
        ([1.0, 0.84, 0.0], "gold"),
        ([0.0, 0.5, 0.0], "dark_green"),
        ([0.0, 0.0, 0.5], "navy"),
        ([0.5, 0.0, 0.0], "maroon"),
        ([0.0, 0.5, 0.5], "teal"),
        ([0.75, 0.75, 0.75], "silver"),
        ([1.0, 0.27, 0.0], "red_orange"),
        ([0.5, 1.0, 0.5], "light_green"),
        ([0.5, 0.5, 1.0], "light_blue"),
        ([1.0, 0.5, 0.5], "light_red"),
        ([0.8, 0.8, 0.0], "olive"),
        ([0.4, 0.0, 0.4], "dark_purple"),
        ([0.0, 0.4, 0.4], "dark_teal"),
        ([0.8, 0.4, 0.0], "dark_orange"),
        ([0.6, 0.8, 1.0], "sky_blue"),
        ([1.0, 0.6, 0.8], "hot_pink"),
        ([0.4, 0.8, 0.4], "lime_green"),
        ([0.8, 0.6, 0.4], "tan"),
        ([0.2, 0.2, 0.8], "ultramarine")
    ]
    
    if print_color_mapping:
        print("\n=== Object ID to Color Mapping ===")
    
    for i, obj in enumerate(bbq_objects):

        if print_edges:
            vl_sat_edges = obj.get('edges_vl_sat', [])
            # {'id_1': 2, 'class_name_1': 'class 2', 'rel_name': 'bigger than', 'id_2': 0, 'class_name_2': 'class 0', 'rel_id': 3
            for edge in vl_sat_edges:
                print(f"vl_sat edge: {edge['id_1']} -- {edge['rel_name']} --> {edge['id_2']}")
            
            sv_edges = obj.get('edges_sv', [])
            # {'source': 0, 'target': -2, 'relation': 'hanging on'}
            for edge in sv_edges:
                print(f"sv edge: {edge['source']} -- {edge['relation']} --> {edge['target']}")

        # Determine a displayable object id string
        raw_id = obj.get('node_id', i)
        if isinstance(raw_id, (list, tuple, np.ndarray)):
            obj_id_str = '_'.join(str(x) for x in list(raw_id)[:3])
        else:
            obj_id_str = str(raw_id)

        # Check if the object dictionary contains the point cloud key
        if 'pcd_np' not in obj or not isinstance(obj['pcd_np'], np.ndarray):
            print(f"Warning: Skipping object ID {obj_id_str} because it's missing a valid 'pcd_np' point cloud.")
            continue
            
        pcd_np = obj['pcd_np']
        
        # Check if the point cloud is empty
        if pcd_np.shape[0] == 0:
            print(f"Warning: Skipping object ID {obj_id_str} because its point cloud is empty.")
            continue
            
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd_np)
        
        # Assign color from palette based on index
        color_rgb, color_name = color_palette[i % len(color_palette)]
        pcd_o3d.paint_uniform_color(color_rgb)
        
        # Print the mapping
        if print_color_mapping:
            print(f"Object ID {obj_id_str} -> {color_name}")
        
        geometries.append(pcd_o3d)
        
        # Blue color for all bboxes
        blue_bbox_color = [0.0, 0.0, 1.0]

        # --- Bounding box drawing (only if show_bbox is True) ---
        if show_bbox:
            bbox_np = None
            obb = None
            aabb = None
            if 'bbox_np' in obj and isinstance(obj['bbox_np'], np.ndarray) and obj['bbox_np'].size:
                try:
                    bbox_np = obj['bbox_np']
                    # Prefer oriented bounding box created from the 8 corner points to preserve rotation
                    obb = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(bbox_np))
                    obb.color = blue_bbox_color
                    geometries.append(obb)
                except Exception as e:
                    print(f"Warning: Failed to create OBB for object {obj_id_str} from 'bbox_np': {e}")

            # If center+extent+rotation exist, draw oriented box in blue (keeping earlier logic)
            if 'bbox_center' in obj and 'bbox_extent' in obj and 'bbox_rotation' in obj:
                try:
                    center_ce = np.array(obj['bbox_center'], dtype=float)
                    extent_ce = np.array(obj['bbox_extent'], dtype=float)
                    R_ce = np.array(obj['bbox_rotation'], dtype=float)
                    aabb = o3d.geometry.OrientedBoundingBox(center_ce, R_ce, extent_ce)
                    aabb.color = blue_bbox_color
                    geometries.append(aabb)
                except Exception as e:
                    print(f"Warning: Failed to create OBB(center/extent/rot) for object {obj_id_str}: {e}")

        # --- Center drawing (only if show_center is True) ---
        if show_center:
            center = None
            if 'bbox_center' in obj:
                try:
                    center = np.array(obj['bbox_center'], dtype=float)
                except Exception:
                    center = None
            elif show_bbox and 'bbox_np' in obj and isinstance(obj['bbox_np'], np.ndarray) and obj['bbox_np'].size:
                bbox_np = obj['bbox_np']
                center = bbox_np.mean(axis=0)

            radius = 0.02
            if center is not None:
                try:
                    if show_bbox and 'bbox_np' in obj and isinstance(obj['bbox_np'], np.ndarray) and obj['bbox_np'].size:
                        bbox_np = obj['bbox_np']
                        min_pt = bbox_np.min(axis=0)
                        max_pt = bbox_np.max(axis=0)
                        ext = max_pt - min_pt
                        radius = max(0.01, float(np.max(ext)) * 0.04)
                    elif 'bbox_extent' in obj:
                        ext = np.array(obj['bbox_extent'], dtype=float)
                        radius = max(0.01, float(np.max(ext)) * 0.04)
                except Exception:
                    pass
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
                sphere.compute_vertex_normals()
                sphere.paint_uniform_color([1.0, 1.0, 0.0])  # yellow center
                sphere.translate(center)
                geometries.append(sphere)

                # --- Text label above center ---
                label_offset = np.array([0.0, 0.0, radius * 2.5])  # assumes Z-up
                label_pos = center + label_offset
                text_mesh = _create_text_label(obj_id_str, label_pos, color=(1,1,1))
                if text_mesh is not None:
                    geometries.append(text_mesh)
                else:
                    # Fallback: create a small line (billboard surrogate) if text unsupported
                    try:
                        line_pts = [center + label_offset, center + label_offset + np.array([0,0,radius*0.5])]
                        line_set = o3d.geometry.LineSet()
                        line_set.points = o3d.utility.Vector3dVector(np.vstack(line_pts))
                        line_set.lines = o3d.utility.Vector2iVector([[0,1]])
                        line_set.colors = o3d.utility.Vector3dVector([[1,1,1]])
                        geometries.append(line_set)
                    except Exception:
                        pass
    
    if not geometries:
        print("No valid point clouds were found to visualize.")
        return
        
    o3d.visualization.draw_geometries(geometries, window_name="BBQ Final Scene Visualization")

if __name__ == "__main__":
    # Import matplotlib here only if needed for the color map
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: Matplotlib not found. Using random colors for visualization.")
        plt = None # Set to None if not available

    
    # input_file = "/home/rizo/mipt_ccm/warehouse/code_pack/BeyondBareQueries/output/frame_last_objects.pkl.gz"
    # input_file = "/home/rizo/mipt_ccm/bbq/08.26.2025_22:32:40_isaac_warehouse_objects.pkl.gz"
    # input_file = "/home/rizo/mipt_ccm/bbq/GRASSETSANDMODELS/scene/scene_test/2.pkl.gz"
    # input_file = "/home/rizo/mipt_ccm/bbq/08.27.2025_14:25:15_isaac_warehouse_objects.pkl.gz"
    input_file = "/home/rizo/mipt_ccm/bbq/08.27.2025_16:24:41_isaac_warehouse_objects.pkl.gz"
    input_file = "/home/rizo/mipt_ccm/bbq/bbq_results/warehouse12_str1.pkl.gz"
    # --- Load the data ---
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        exit()
        
    print(f"Loading BBQ scene from {input_file}...")
    
    try:
        with gzip.open(input_file, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error: Failed to load or decompress the pickle file. Reason: {e}")
        exit()

    # --- Validate the data structure and extract objects ---
    if not isinstance(data, dict) or 'objects' not in data:
        print("Error: The pickle file does not have the expected format. It should be a dictionary with an 'objects' key.")
        exit()
    
    print(f"Key in data: {list(data.keys())}")    
    object_list = data['objects']
    
    print(f"Loaded {len(object_list)} objects from the BBQ scene.")
    print(f"Object IDs: {[obj.get('id', i) for i, obj in enumerate(object_list)]}")
    
    print(object_list[0].keys())  # Print the first object for inspection
    print(object_list[0]['id'], "ids")  # Print the first object for inspection
    print(object_list[0]['pcd_np'].shape, "pcd_np shape")  # Print the shape of the point cloud for the first object
    print(object_list[0]['bbox_np'].shape, "bbox_np shape")  # Print the shape of the bounding box for the first object
    print(object_list[0]['pcd_color_np'].shape, "pcd_color_shape")  # Print the shape of the point cloud color for the first object
    # print(object_list[0]['bbox_centqnt'].shape, object_list[0]['bbox_extent'], "bbox_extent")  # Print the shape of the point cloud color for the first object
    # print(object_list[0]['color_image_idx'], "color img idx")  # Print the pose for the first object
    # for obj in object_list:
    #     print(f"color index list: {obj.get('color_image_idx', 'N/A')}")
    
    # --- Visualize the data ---
    visualize_bbq_objects(object_list)
