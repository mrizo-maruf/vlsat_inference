import os
import json
import numpy as np
import copy
import torch
from itertools import product
from collections import Counter
import open3d as o3d
import torch.nn.functional as F
import gzip
import pickle
if __name__ == '__main__':
    os.sys.path.append('./src')
from src.utils.config import Config
from src.model.model import MMGNet
from src.utils import op_utils
import random


def bbox_center_from_bbox_np(bbox_np):
    """
    bbox_np: (N,3) numpy array of bounding box corner points (e.g. 8 corners).
    Returns center (3,)
    """
    pts = np.asarray(bbox_np)
    return pts.mean(axis=0)

class EdgePredictor:
    def __init__(self, config_path, ckpt_path, relationships_list):
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.config = Config(config_path)
        self.config.exp = ckpt_path
        self.config.MODE = "eval"
        self.padding = 0.2
        self.model = MMGNet(self.config)
        # init device
        if torch.cuda.is_available() and len(self.config.GPU) > 0:
            self.config.DEVICE = torch.device("cuda")
        else:
            self.config.DEVICE = torch.device("cpu")
        self.model.load(best=True)
        self.model.model.eval()
        
        with open(relationships_list, "r") as f:
            self.relationships_list = f.readlines()
        
        self.rel_id_to_rel_name = {
            i: name.strip()
            for i, name in enumerate(self.relationships_list[1:])
        }
        
        print(f"Loaded {len(self.relationships_list)} relationships from {relationships_list}.")
        print("relationships_list (first 10):", self.relationships_list[:10])
        
        print(f"Edge Predictor initialized for device '{self.config.DEVICE}'.")

    def preprocess_poinclouds(self, points, num_points):
        assert len(points) > 1, "Number of objects should be at least 2"
        print(num_points, "num_points")
        
        # print("points shape:", [p.shape for p in points])
        # print("points length:", len(points))
        # print("points[0] shape:", points[0].shape)
        # print("points[0] first 5 points:", points[0][:5])
        # print("points[-1] first 5 points:", points[-1][:5])
        
        edge_indices = list(product(list(range(len(points))), list(range(len(points)))))
        edge_indices = [i for i in edge_indices if i[0]!=i[1]]

        num_objects = len(points)
        dim_point = points[0].shape[-1]

        instances_box = dict()
        print("num_objects:", num_objects, "dim_point:", dim_point, "num_points:", num_points)
        obj_points = torch.zeros([num_objects, num_points, dim_point])
        descriptor = torch.zeros([num_objects, 11])

        obj_2d_feats = np.zeros([num_objects, 512])

        for i, pcd in enumerate(points):
            # get node point
            
            min_box = np.min(pcd, 0) - self.padding
            max_box = np.max(pcd, 0) + self.padding
            
            instances_box[i] = (min_box, max_box)
            
            choice = np.random.choice(len(pcd), num_points, replace=True)
            print(f"choice shape: {choice.shape}, choice first 5: {choice[:5]}")
            
            pcd = pcd[choice, :]
            descriptor[i] = op_utils.gen_descriptor(torch.from_numpy(pcd))
            pcd = torch.from_numpy(pcd.astype(np.float32))
            pcd = self.zero_mean(pcd)
            
            obj_points[i] = pcd

        edge_indices = torch.tensor(edge_indices, dtype=torch.long).permute(1, 0)
        obj_2d_feats = torch.from_numpy(obj_2d_feats.astype(np.float32))    
        obj_points = obj_points.permute(0, 2, 1)
        batch_ids = torch.zeros((num_objects, 1))
        return obj_points, obj_2d_feats, edge_indices, descriptor, batch_ids

    def predict_relations(self, obj_points, obj_2d_feats, edge_indices, descriptor, batch_ids):
        print("obj_points shape:", obj_points.shape, "obj_points:", obj_points[-1, :, 0:5])
        
        obj_points = obj_points.to(self.config.DEVICE)
        obj_2d_feats = obj_2d_feats.to(self.config.DEVICE)
        edge_indices = edge_indices.to(self.config.DEVICE)
        descriptor = descriptor.to(self.config.DEVICE)
        batch_ids = batch_ids.to(self.config.DEVICE)
        
        # print("obj_points shape:", obj_points.shape, "obj_points:", obj_points[-1, :, 0:5])
        # print("obj_2d_feats shape:", obj_2d_feats.shape, "obj_2d_feats:", obj_2d_feats[0])
        # print("edge_indices shape:", edge_indices.shape, "edge_indices:", edge_indices[0])
        # print("descriptor shape:", descriptor.shape, "descriptor0:", descriptor[0])
        # print("batch_ids shape:", batch_ids.shape, "batch_ids:", batch_ids)
        
        with torch.no_grad():
            rel_cls_3d = self.model.model(
                obj_points, obj_2d_feats, edge_indices, descriptor, batch_ids=batch_ids
            )
        return rel_cls_3d

    def save_relations(self, tracking_ids, timestamps, class_names, predicted_relations, edge_indices):
        saved_relations = []
        for k in range(predicted_relations.shape[0]):
            idx_1 = edge_indices[0][k].item()
            idx_2 = edge_indices[1][k].item()

            id_1 = tracking_ids[idx_1]
            id_2 = tracking_ids[idx_2]

            timestamp_1 = timestamps[idx_1]
            timestamp_2 = timestamps[idx_2]

            class_name_1 = class_names[idx_1]
            class_name_2 = class_names[idx_2]

            rel_id = torch.argmax(predicted_relations, dim=1)[k].item()
            rel_name = self.rel_id_to_rel_name[rel_id]

            rel_dict = {
                #"id_1": id_1,
                #"timestamp_1": timestamp_1,
                "class_name_1": class_name_1,
                "rel_name": rel_name,
                #"id_2": id_2,
                #"timestamp_2": timestamp_2,
                "class_name_2": class_name_2,
                #"rel_id": rel_id,
                
            }
            saved_relations.append(rel_dict)

        return saved_relations

    def zero_mean(self, point):
        mean = torch.mean(point, dim=0)
        point -= mean.unsqueeze(0)
        return point

def pcd_denoise_dbscan(pcd: o3d.geometry.PointCloud, eps=0.02, min_points=10) -> o3d.geometry.PointCloud:
    ### Remove noise via clustering
    pcd_clusters = pcd.cluster_dbscan(
        eps=eps,
        min_points=min_points,
    )
    
    # Convert to numpy arrays
    obj_points = np.asarray(pcd.points)
    #obj_colors = np.asarray(pcd.colors)
    pcd_clusters = np.array(pcd_clusters)

    # Count all labels in the cluster
    counter = Counter(pcd_clusters)

    # Remove the noise label
    if counter and (-1 in counter):
        del counter[-1]

    if counter:
        # Find the label of the largest cluster
        most_common_label, _ = counter.most_common(1)[0]
        
        # Create mask for points in the largest cluster
        largest_mask = pcd_clusters == most_common_label

        # Apply mask
        largest_cluster_points = obj_points[largest_mask]
        #largest_cluster_colors = obj_colors[largest_mask]
        
        # If the largest cluster is too small, return the original point cloud
        if len(largest_cluster_points) < 5:
            return pcd

        # Create a new PointCloud object
        largest_cluster_pcd = o3d.geometry.PointCloud()
        largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)
        
        pcd = largest_cluster_pcd
        
    return pcd

def main():
    config_path = "config/mmgnet.json"
    ckpt_path = "/home/rizo/mipt_ccm/vlsat_inference/3dsgg_best_chkp/3dssg_best_ckpt"
    data_path = "./point_clouds"
    relationships_list = "/home/rizo/mipt_ccm/vlsat_inference/data/3DSSG_subset/relations.txt"

    pcds = {}
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: Matplotlib not found. Using random colors for visualization.")
        plt = None # Set to None if not available

    
    input_file = "/home/rizo/mipt_ccm/warehouse/code_pack/BeyondBareQueries/output/frame_last2_objects.pkl.gz"
    input_file = "/home/rizo/mipt_ccm/warehouse/code_pack/BeyondBareQueries/output/scenes/08.17.2025_23:35:41_isaac_warehouse_objects.pkl.gz"

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
        
    object_list = data['objects']
    pcds = {}
    for obj_id, obj in enumerate(object_list):
        pcds[obj_id] = {}
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(obj['pcd_np'])
        pcd = pcd_denoise_dbscan(pcd_o3d)
        
        # TO-DO: Add pose information if available
        center = bbox_center_from_bbox_np(obj['bbox_np'])
        obj['pose'] = center.tolist() 
        pose = obj["pose"]
        
        # random
        # pose = [random.randint(10, 50) for _ in range(3)]
        

        pcd_array = np.array(pcd.points)
        
        size = [0.3, 0.3, 0.15]
        nx, ny, nz = (16, 16, 16)
        
        x = np.linspace(pose[0] - size[0]/2, pose[0] + size[0]/2, nx)
        y = np.linspace(pose[1] - size[1]/2, pose[1] + size[1]/2, ny)
        z = np.linspace(pose[2] - size[2]/2, pose[2] + size[2]/2, nz)
        xv, yv, zv = np.meshgrid(x, y, z)
        
        # print(x.shape, y.shape, z.shape)
        # print(xv)
        # print(xv.shape, yv.shape, zv.shape)
        #print(xv.flatten())
        
        grid_pc = np.stack((xv.flatten(), yv.flatten(), zv.flatten()), axis=1)
        pcds[obj_id]['point_cloud'] = grid_pc
        
        pcds[obj_id]['position'] = [
                   np.round(np.mean(pcds[obj_id]['point_cloud'][:, 0]),2),
                   np.round(np.mean(pcds[obj_id]['point_cloud'][:, 1]),2),
                   np.round(np.mean(pcds[obj_id]['point_cloud'][:, 2]),2),
                ]
        
    # print("Loaded the following saved pointclouds:")
    # for obj_id, obj_pcds in pcds.items():
    #     print(obj_id, "with position ", obj_pcds['position'], "point cloud shape", obj_pcds['point_cloud'].shape)
        
    print(f"pcds keys: {list(pcds.keys())}")
    pcds_list = [pcd['point_cloud'] for pcd in pcds.values()]
    print(f"pcds_list length: {len(pcds_list)}, type: {type(pcds_list[0])}")
    print(f"pcds_list shapes: {[pcd.shape for pcd in pcds_list]}")
    
    #exit()
    edge_predictor = EdgePredictor(config_path, ckpt_path, relationships_list)

    obj_points, obj_2d_feats, edge_indices, descriptor, batch_ids = edge_predictor.preprocess_poinclouds(
        pcds_list,
        edge_predictor.config.dataset.num_points
    )
    predicted_relations = edge_predictor.predict_relations(obj_points, obj_2d_feats, edge_indices, descriptor, batch_ids)
    print(f"predicted_relations shape: {predicted_relations.shape}, type: {type(predicted_relations)}")
    print(f"predicted_relations first 5: {predicted_relations[:5]}")
    
    #print(predicted_relations.shape)
    topk_values, topk_indices = torch.topk(predicted_relations, 5, dim=1,  largest=True)
    # print(topk_indices, topk_values)
    
    tracking_ids = [str(i) for i in range(len(pcds_list))]
    timestamps = ["001539" for i in range(len(pcds_list))]
    class_names = [f"class {i}" for i in range(len(pcds_list))]

    saved_relations = edge_predictor.save_relations(tracking_ids, timestamps, class_names, predicted_relations, edge_indices)

    # print("Predicted the following relations:")
    # print(json.dumps(saved_relations, indent=4))
    
    with open("saved_relations.json", "w") as f:
        json.dump(saved_relations, f, indent=4)

    print("Saved to saved_relations.json")

    return saved_relations
    # return


if __name__ == "__main__":
    main() 
