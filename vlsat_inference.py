import os
import numpy as np
import copy
import torch

import open3d as o3d

if __name__ == '__main__':
    os.sys.path.append('./src')
from src.utils.config import Config
from src.model.model import MMGNet


class EdgePredictor:
    def __init__(self, config_path, ckpt_path):
        self.config = Config(config_path)
        self.config.exp = ckpt_path
        self.config.MODE = "eval"
        self.model = MMGNet(self.config)
        # init device
        if torch.cuda.is_available() and len(self.config.GPU) > 0:
            self.config.DEVICE = torch.device("cuda")
        else:
            self.config.DEVICE = torch.device("cpu")

        self.model.load(best=True)
        
    def preprocess_poinclouds():
        pass
    
    def predict_relations():
        pass

    def save_relations():
        pass

def main():
    config_path = "config/mmgnet.json"
    ckpt_path = "/hdd/wingrune/3dssg_best_ckpt"
    data_path = "./point_clouds"
    pcds = {}
    for pcd_path in os.listdir(data_path):
        _, _, _, timecode, _, obj_id = pcd_path.split(".")[0].split("_")
        if obj_id not in pcds:
            pcds[obj_id] = {}
        data_dict = np.load(os.path.join(data_path, pcd_path), allow_pickle=True).item()
        pcds[obj_id][timecode] = {}
        pcds[obj_id][timecode]['point_cloud'] = copy.deepcopy(data_dict['point_cloud'])
        pcds[obj_id][timecode]['position'] = [
            np.round(np.mean(pcds[obj_id][timecode]['point_cloud'][:,0]),2),
            np.round(np.mean(pcds[obj_id][timecode]['point_cloud'][:,1]),2),
            np.round(np.mean(pcds[obj_id][timecode]['point_cloud'][:,2]),2)
        ]
    
    for obj_id, obj_pcds in pcds.items():
        for timecode in obj_pcds:
            print(obj_id, timecode, obj_pcds[timecode]['position'])
    
    edge_predictor = EdgePredictor(config_path, ckpt_path)
    input()
    preprocessed_pcds = edge_predictor.preprocess_poinclouds(pcds)
    predicted_relations = edge_predictor.predict_relations(preprocessed_pcds)
    edge_predictor.save_relations(predicted_relations)
    print("hey there") 

if __name__=="__main__":
    """
    data_path = "./point_clouds"
    pcd_path = "point_cloud_20240806_001539_id_7.npy"
    data_dict = np.load(os.path.join(data_path, pcd_path), allow_pickle=True).item()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data_dict["point_cloud"])
    o3d.visualization.draw_geometries([pcd])
    input()
    pcd_path = "point_cloud_20240806_001540_id_7.npy"
    data_dict = np.load(os.path.join(data_path, pcd_path), allow_pickle=True).item()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data_dict["point_cloud"])
    o3d.visualization.draw_geometries([pcd])
    """
    main() 
