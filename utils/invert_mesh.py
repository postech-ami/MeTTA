import open3d as o3d
import numpy as np
import os
from glob import glob

def invert_face_orientations(mesh):
    # 메쉬의 트라이앵글과 정점을 가져옵니다.
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)

    # 트라이앵글의 방향을 반전시키고 새로운 트라이앵글의 리스트를 만듭니다.
    inverted_triangles = triangles[:, [0, 2, 1]]

    # 새로운 트라이앵글로부터 트라이앵글 노말을 계산합니다.
    inverted_triangle_normals = np.asarray(mesh.triangle_normals) * -1

    # 새로운 트라이앵글 메쉬를 생성합니다.
    inverted_mesh = o3d.geometry.TriangleMesh()
    inverted_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    inverted_mesh.triangles = o3d.utility.Vector3iVector(inverted_triangles)
    inverted_mesh.triangle_normals = o3d.utility.Vector3dVector(inverted_triangle_normals)

    return inverted_mesh

# 예시로 메쉬를 불러오고 방향을 반전시킵니다.
if __name__ == "__main__":
    dir_names = ["3d_front", "ohouse", "pix3d", "pix3d_im3d", "pix3d_im3d_extended", "real_chair", "real_im3d", "real_table", "rebuttal", "teaser"]
    root_path = "/node_data/ugkim/sitto/src/"
    for dir_name in dir_names:
        print(f"[INFO] current dir name: {dir_name}")
        sub_dir_list = os.listdir(f"/node_data/ugkim/sitto/src/{dir_name}")
        sub_dir_list = [sub for sub in sub_dir_list if os.path.isdir(os.path.join(root_path, dir_name, sub))]
        for sub in sub_dir_list:
            before_dir = f"/node_data/ugkim/sitto/src/{dir_name}/{sub}/mesh"
            os.rename(f"/node_data/ugkim/sitto/src/{dir_name}/{sub}/mesh", f"/node_data/ugkim/sitto/src/{dir_name}/{sub}/mesh_old")
            os.makedirs(before_dir, exist_ok=True)
            mesh_list = glob(f"/node_data/ugkim/sitto/src/{dir_name}/{sub}/mesh_old/*.obj")
            for mesh_name in mesh_list:
                mesh = o3d.io.read_triangle_mesh(mesh_name)
                inverted_mesh = invert_face_orientations(mesh)
                obj_name = mesh_name.split("/")[-1]
                new_mesh_name = os.path.join(f"/node_data/ugkim/sitto/src/{dir_name}/{sub}/mesh/{obj_name}")
                o3d.io.write_triangle_mesh(new_mesh_name, inverted_mesh)

    print("[INFO] invert mesh is done.")