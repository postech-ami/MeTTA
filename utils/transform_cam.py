import os
import math
import numpy as np
import json
from pprint import pprint

import torch
# import pytorch3d.transforms as T

# # Mesh R-CNN
# def meshrcnn_radians(rotation_path):
#     with open(rotation_path, "r") as json_file:
#         pix3d = json.load(json_file)
    
#     pix3d_info = None
#     for i in range(len(pix3d)):
#         if pix3d[i]['img'] == "img/chair/0169.jpg":
#             pix3d_info = pix3d[i]
#             break
#     rot_matrix = pix3d_info["rot_mat"]
#     rot_matrix = torch.tensor(rot_matrix)
#     inv_rot_matrix = torch.linalg.inv(rot_matrix)

    
#     angle = T.matrix_to_euler_angles(inv_rot_matrix, "YXZ")
#     return angle[0], angle[1]  # azimuth, elevation

# Total3D
def get_radians(src_path, pix3d_class, pix3d_id):
    json_path = f"{src_path}/annotation.json"
    # prefix = f"{src_path}/"
    with open(json_path, "r") as json_file:
        json_file = json.load(json_file)
        for i in range(len(json_file)):
            
            # find category
            if json_file[i]['category'] == pix3d_class:
                instances = json_file[i]['instances']
                
                # find instances
                for j in range(len(instances)):
                    if instances[j]["id"] == pix3d_id:
                        azimuth = instances[j]['azimuth']
                        elevation = instances[j]['elevation']
                        # mesh_path = prefix + instances[j]['mesh']
                        mesh_path = os.path.join(src_path, instances[j]['mesh'])
                        image_path = instances[j]['image']
                        # image_path = prefix + image_path.split(".")[0] + "_rgba.png"
                        image_path = os.path.join(src_path, image_path.split(".")[0] + "_rgba.png")
    
    return torch.tensor(azimuth), torch.tensor(elevation), mesh_path, image_path  # azimuth, elevation, mesh, image path

def get_radians_noise(src_path, pix3d_class, pix3d_id, noise_level, noise_type):
    json_path = f"{src_path}/annotation.json"
    # prefix = f"{src_path}/"
    with open(json_path, "r") as json_file:
        json_file = json.load(json_file)
        for i in range(len(json_file)):
            
            # find category
            if json_file[i]['category'] == pix3d_class:
                instances = json_file[i]['instances']
                
                # find instances
                for j in range(len(instances)):
                    if instances[j]["id"] == pix3d_id:
                        azimuth = instances[j]['azimuth']
                        elevation = instances[j]['elevation']
                        # mesh_path = prefix + instances[j]['mesh']
                        mesh_path = os.path.join(src_path, instances[j]['mesh'])
                        image_path = instances[j]['image']
                        # image_path = prefix + image_path.split(".")[0] + "_rgba.png"
                        image_path = os.path.join(src_path, image_path.split(".")[0] + "_rgba.png")
    
    if noise_type == "azimuth":
        azimuth = azimuth + noise_level
    elif noise_type == "elevation":
        elevation = elevation + noise_level
    elif noise_type == "both":
        azimuth = azimuth + noise_level
        elevation = elevation + noise_level
    return torch.tensor(azimuth), torch.tensor(elevation), mesh_path, image_path  # azimuth, elevation, mesh, image path

if __name__ == "__main__":
    # print("[For Mesh R-CNN]")
    # azimuth, elevation = meshrcnn_radians("dataset/pix3d.json")
    # print(f"azimuth: {torch.rad2deg(azimuth)}")
    # print(f"elevation: {torch.rad2deg(elevation)}")

    print("[For Total3D, Im3D]")
    src_path = os.path.join("/node_data/ugkim/sitto", "src", "pix3d_im3d")
    azimuth, elevation, mesh_path, image_path = get_radians(src_path=src_path, pix3d_class="desk", pix3d_id="0245")
    print(f"azimuth: {np.rad2deg(azimuth)}")
    print(f"elevation: {np.rad2deg(elevation)}")

    azimuth, elevation, mesh_path, image_path = get_radians_noise(src_path=src_path, pix3d_class="desk", pix3d_id="0245", noise_level=0.1)