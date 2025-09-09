#!/usr/bin/env python3
"""
Script to remove objects with specified node IDs from a BBQ pickle file and save the filtered result.

Usage:
    python remove_objects_by_id.py input.pkl.gz output.pkl.gz --remove-ids 1 2 3
    python remove_objects_by_id.py input.pkl.gz output.pkl.gz --remove-ids 5 --dry-run
"""

import argparse
import gzip
import pickle
import os
import sys
from typing import List, Union


def load_bbq_data(input_file: str) -> dict:
    """Load BBQ data from a gzipped pickle file."""
    print(f"Loading BBQ scene from {input_file}...")
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    try:
        with gzip.open(input_file, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load or decompress the pickle file: {e}")
    
    if not isinstance(data, dict) or 'objects' not in data:
        raise ValueError("The pickle file does not have the expected format. It should be a dictionary with an 'objects' key.")
    
    return data


def save_bbq_data(data: dict, output_file: str) -> None:
    """Save BBQ data to a gzipped pickle file."""
    print(f"Saving filtered BBQ scene to {output_file}...")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        with gzip.open(output_file, "wb") as f:
            pickle.dump(data, f)
    except Exception as e:
        raise RuntimeError(f"Failed to save the pickle file: {e}")


def get_object_id(obj: dict, index: int) -> Union[str, int]:
    """Extract object ID from object dictionary."""
    # Try different possible ID keys
    for id_key in ['node_id', 'id']:
        if id_key in obj:
            raw_id = obj[id_key]
            if isinstance(raw_id, (list, tuple)):
                # If it's a list/tuple, use the first element or join them
                return raw_id[0] if len(raw_id) == 1 else '_'.join(str(x) for x in raw_id[:3])
            else:
                return raw_id
    
    # Fallback to index if no ID found
    return index


def remove_objects_by_ids(data: dict, remove_ids) -> dict:
    """Remove objects with specified IDs from the data and clean up their edges."""
    original_count = len(data)
    
    print(f"\nOriginal object count: {original_count}")
    print(f"IDs to remove: {remove_ids}")
    
    # Filter objects
    filtered_objects = []
    removed_objects = []
    
    clean_data = [obj.copy() for obj in data if obj['node_id'] not in remove_ids]
    
    for obj in clean_data:
        obj['edges_vl_sat'], obj['edges_sv'] = clean_object_edges(obj, remove_ids)
    # # remove nodes
    # try: 
    #     print("DEBUG: Starting node removal process")
    #     for obj in data:
    #         obj_id = obj['node_id']
            
    #         if obj_id not in remove_ids:
    #             print(f"DEBUG: {obj_id} is in remove list")
    #             print(f"DEBUG: Removing object with ID {obj_id} == pop id: {data[obj_id]['node_id']}")
    #             data = data.pop(obj_id)
    # except Exception as e:
    #     print(f"Error during node removal: {e}")

    # print(f"DEBUG: Starting edge removal process")
    # # remove edges
    # for obj in data:
    #     obj_id = obj['node_id']
        
    #     if obj_id in remove_ids:
    #         print(f"DEBUG: {obj_id} is in remove list")
        
    #         # Keep this object but clean up its edges
    #         clean_object_edges(obj, remove_ids)
            
    return clean_data


def clean_object_edges(obj: dict, remove_ids_set: set) -> dict:
    """Remove edges that reference deleted object IDs from an object."""
    edges_cleaned = 0
    
    clean_vl_sat = [edge for edge in obj['edges_vl_sat'] if edge['id_2'] not in remove_ids_set]
    clean_sv = [edge for edge in obj['edges_sv'] if edge['target'] not in remove_ids_set]
    
    return clean_vl_sat, clean_sv

def main():
    try:
        input_file = "/home/rizo/mipt_ccm/bbq/bbq_results/scenes_scene_3ch_1t/scene_3ch_1t_str1.pkl.gz"
        output_file = "/home/rizo/mipt_ccm/bbq/bbq_results/scenes_scene_3ch_1t/scene_3ch_1t_str1_cleaned.pkl.gz"
        # Load data
        data = load_bbq_data(input_file)
        
        # Show original object IDs
        object_list = data['objects']
        
        # scene_01_big_table_str1: [10, 6, 11, 5, 14]
        
        # scene_02_small_table_str1: [5, 6, 9, 11, 12]
        
        # scene_3ch_1t_str1: [2, 4, 3, 8, 10]
        remove_ids = [0, 1, 5, 6, 7, 8, 9, 11, 12]
        
        # Remove specified objects
        filtered_data = remove_objects_by_ids(object_list, remove_ids)
        
        data['objects'] = filtered_data
        
        # Save filtered data
        save_bbq_data(data, output_file)
        print(f"\nFiltered data saved successfully to {output_file}")
        
        for obj in filtered_data:
            print(f"Remaining object ID: {obj['node_id']}")
            for edge in obj['edges_vl_sat']:
                print(f"  Edge to ID: {edge['id_1']} -> {edge['rel_name']} -> {edge['id_2']}")
                
        # for obj in filtered_data:
        #     print(f"Remaining object ID: {obj['node_id']}")
        #     for edge in obj['edges_sv']:
        #         print(f"  Edge to ID: {edge['source']} -> {edge['relation']} -> {edge['target']}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
