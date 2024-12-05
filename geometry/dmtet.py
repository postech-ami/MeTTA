# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import numpy as np
import torch

from render import mesh
from render import render
from render import regularizer

from encoding import get_encoder

import tqdm
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
import os
import wandb
import cv2

from gridencoder import GridEncoder

import open3d as o3d
from render import util

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []

        if False:
            for l in range(num_layers):
                net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=bias))
                # net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=True))  # TODO: bias false -> true

        if True:  # CHECK: need to fix flip-normal debugging
            for l in range(num_layers-1):  # TODO: bias False -> True?
                # net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=bias))
                net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=False))
            l = l + 1
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=True))
            

        self.net = nn.ModuleList(net)

        # by Fantasia3D author, initialize is important
        self.net.apply(self._init_weights)
    
    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                if True:
                    x = F.relu(x, inplace=True)
                if False:  # CHECK: flip-normal debugging
                    x = F.relu(x)
        return x
    
    @staticmethod
    def _init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0.0)


def torch_vis_2d(x, renormalize=False):
    # x: [3, H, W] or [1, H, W] or [H, W]
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    
    if isinstance(x, torch.Tensor):
        if len(x.shape) == 3:
            x = x.permute(1,2,0).squeeze()
        x = x.detach().cpu().numpy()
        
    print(f'[torch_vis_2d] {x.shape}, {x.dtype}, {x.min()} ~ {x.max()}')
    
    x = x.astype(np.float32)
    
    # renormalize
    if renormalize:
        x = (x - x.min(axis=0, keepdims=True)) / (x.max(axis=0, keepdims=True) - x.min(axis=0, keepdims=True) + 1e-8)

    plt.imshow(x)
    plt.show()

###############################################################################
# Marching tetrahedrons implementation (differentiable), adapted from
# https://github.com/NVIDIAGameWorks/kaolin/blob/master/kaolin/ops/conversions/tetmesh.py
###############################################################################

# issue: https://github.com/NVlabs/nvdiffrec/issues/131
class DMTet:
    def __init__(self):
        self.triangle_table = torch.tensor([
                [-1, -1, -1, -1, -1, -1],
                [ 1,  0,  2, -1, -1, -1],
                [ 4,  0,  3, -1, -1, -1],
                [ 1,  4,  2,  1,  3,  4],
                [ 3,  1,  5, -1, -1, -1],
                [ 2,  3,  0,  2,  5,  3],
                [ 1,  4,  0,  1,  5,  4],
                [ 4,  2,  5, -1, -1, -1],
                [ 4,  5,  2, -1, -1, -1],
                [ 4,  1,  0,  4,  5,  1],
                [ 3,  2,  0,  3,  5,  2],
                [ 1,  3,  5, -1, -1, -1],
                [ 4,  1,  2,  4,  3,  1],
                [ 3,  0,  4, -1, -1, -1],
                [ 2,  0,  1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1]
                ], dtype=torch.long, device='cuda')  # 16 x 6 (6 is num of edges), all tet-edges cases, -1 indicates N/A

        self.num_triangles_table = torch.tensor([0,1,1,2,1,2,2,1,1,2,2,1,2,1,1,0], dtype=torch.long, device='cuda')  # 16 types triangles or quds from tetreheron. 2 means two triangles, i.e., one quad.
        self.base_tet_edges = torch.tensor([0,1, 0,2 ,0,3 ,1,2 ,1,3, 2,3], dtype=torch.long, device='cuda')  # total 6 edges

    ###############################################################################
    # Utility functions
    ###############################################################################

    def sort_edges(self, edges_ex2):
        with torch.no_grad():
            order = (edges_ex2[:,0] > edges_ex2[:,1]).long()
            order = order.unsqueeze(dim=1)

            a = torch.gather(input=edges_ex2, index=order, dim=1)      
            b = torch.gather(input=edges_ex2, index=1-order, dim=1)  

        return torch.stack([a, b],-1)

    def map_uv(self, faces, face_gidx, max_idx):
        N = int(np.ceil(np.sqrt((max_idx+1)//2)))
        tex_y, tex_x = torch.meshgrid(
            torch.linspace(0, 1 - (1 / N), N, dtype=torch.float32, device="cuda"),
            torch.linspace(0, 1 - (1 / N), N, dtype=torch.float32, device="cuda"),
            ) # indexing='ij')

        pad = 0.9 / N

        uvs = torch.stack([
            tex_x      , tex_y,
            tex_x + pad, tex_y,
            tex_x + pad, tex_y + pad,
            tex_x      , tex_y + pad
        ], dim=-1).view(-1, 2)

        def _idx(tet_idx, N):
            x = tet_idx % N
            y = torch.div(tet_idx, N, rounding_mode='trunc')
            return y * N + x

        tet_idx = _idx(torch.div(face_gidx, 2, rounding_mode='trunc'), N)
        tri_idx = face_gidx % 2

        uv_idx = torch.stack((
            tet_idx * 4, tet_idx * 4 + tri_idx + 1, tet_idx * 4 + tri_idx + 2
        ), dim = -1). view(-1, 3)

        return uvs, uv_idx

    ###############################################################################
    # Marching tets implementation
    ###############################################################################

    def __call__(self, pos_nx3, sdf_n, tet_fx4):
        # pos_nx3: [N, 3]
        # sdf_n:   [N]
        # tet_fx4: [F, 4]

        with torch.no_grad():
            occ_n = sdf_n > 0
            occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(-1,4)
            occ_sum = torch.sum(occ_fx4, -1) # [F,]
            valid_tets = (occ_sum>0) & (occ_sum<4)
            occ_sum = occ_sum[valid_tets]

            # find all vertices
            all_edges = tet_fx4[valid_tets][:,self.base_tet_edges].reshape(-1,2)
            all_edges = self.sort_edges(all_edges)
            unique_edges, idx_map = torch.unique(all_edges,dim=0, return_inverse=True)  
            
            unique_edges = unique_edges.long()
            mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1,2).sum(-1) == 1
            mapping = torch.ones((unique_edges.shape[0]), dtype=torch.long, device="cuda") * -1
            mapping[mask_edges] = torch.arange(mask_edges.sum(), dtype=torch.long,device="cuda")
            idx_map = mapping[idx_map] # map edges to verts

            interp_v = unique_edges[mask_edges]
        edges_to_interp = pos_nx3[interp_v.reshape(-1)].reshape(-1,2,3)
        edges_to_interp_sdf = sdf_n[interp_v.reshape(-1)].reshape(-1,2,1)
        edges_to_interp_sdf[:,-1] *= -1

        denominator = edges_to_interp_sdf.sum(1,keepdim = True)

        edges_to_interp_sdf = torch.flip(edges_to_interp_sdf, [1])/denominator
        verts = (edges_to_interp * edges_to_interp_sdf).sum(1)

        idx_map = idx_map.reshape(-1,6)

        v_id = torch.pow(2, torch.arange(4, dtype=torch.long, device="cuda"))
        tetindex = (occ_fx4[valid_tets] * v_id.unsqueeze(0)).sum(-1)
        num_triangles = self.num_triangles_table[tetindex]

        # Generate triangle indices
        faces = torch.cat((
            torch.gather(input=idx_map[num_triangles == 1], dim=1, index=self.triangle_table[tetindex[num_triangles == 1]][:, :3]).reshape(-1,3),
            torch.gather(input=idx_map[num_triangles == 2], dim=1, index=self.triangle_table[tetindex[num_triangles == 2]][:, :6]).reshape(-1,3),
        ), dim=0)

        # Get global face index (static, does not depend on topology)
        num_tets = tet_fx4.shape[0]
        tet_gidx = torch.arange(num_tets, dtype=torch.long, device="cuda")[valid_tets]
        face_gidx = torch.cat((
            tet_gidx[num_triangles == 1]*2,
            torch.stack((tet_gidx[num_triangles == 2]*2, tet_gidx[num_triangles == 2]*2 + 1), dim=-1).view(-1)
        ), dim=0)

        uvs, uv_idx = self.map_uv(faces, face_gidx, num_tets*2)

        return verts, faces, uvs, uv_idx


###############################################################################
#  Geometry interface
###############################################################################


# def get_normalize_mesh(pro_path):
#     mesh = o3d.io.read_triangle_mesh(pro_path)
#     vertices = np.asarray(mesh.vertices)
#     shift = np.mean(vertices,axis=0)  # same with mesh.get_center()
#     scale = np.max(np.linalg.norm(vertices-shift, ord=2, axis=1))
#     vertices = (vertices-shift) / scale
#     # FIXME: remove after debug
#     # print(f"mesh shift: {shift}")
#     # print(f"mesh scale: {scale}")
#     mesh.vertices = o3d.cuda.pybind.utility.Vector3dVector(vertices)
#     return mesh

def get_normalize_mesh(pro_path):
    mesh = o3d.io.read_triangle_mesh(pro_path)
    # Get the bounding box of the mesh
    bbox = mesh.get_axis_aligned_bounding_box()

    # Calculate the scaling factor to fit the bounding box within the desired range
    max_dimension = max(bbox.get_max_bound() - bbox.get_min_bound())
    scale_factor = 1.0 / max_dimension

    # Apply the scaling transformation to the mesh
    mesh.scale(scale_factor * 1.8, center=bbox.get_center())

    return mesh

def get_thicker_mesh(mesh, thickness):
    # Compute vertex normals if not already computed
    mesh.compute_vertex_normals()

    # Specify the thickness you want to add (positive for outward, negative for inward)
    # thickness = 0.05  # Adjust this value as needed

    # Offset each vertex along its normal
    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    offset_vertices = vertices + thickness * normals
    mesh.vertices = o3d.utility.Vector3dVector(offset_vertices)

    # Calculate face normals
    o3d.geometry.TriangleMesh.compute_triangle_normals(mesh)

    # Convert face normals to vertex normals
    mesh.compute_vertex_normals()

    # [Open3D WARNING] Write OBJ can not include triangle normals.
    mesh.triangle_normals = o3d.utility.Vector3dVector([])

    return mesh

def get_flipped_mesh(shape):
    coords = o3d.geometry.TriangleMesh.create_coordinate_frame()
    R = coords.get_rotation_matrix_from_yxz((np.pi, 0, 0))
    shape.rotate(R, center=(0, 0, 0))
    
    return shape

class DMTetGeometry(torch.nn.Module):
    def __init__(self, grid_res, scale, FLAGS):
        super(DMTetGeometry, self).__init__()

        self.FLAGS         = FLAGS
        self.grid_res      = grid_res
        self.marching_tets = DMTet()

        tets = np.load('data/tets/{}_tets.npz'.format(self.grid_res))
        self.verts    = torch.tensor(tets['vertices'], dtype=torch.float32, device='cuda') * scale # for 64/128, [N=36562/277410, 3], in [-0.5, 0.5]^3
        self.indices  = torch.tensor(tets['indices'], dtype=torch.long, device='cuda') # for 64/128, [M=192492/1524684, 4], 4 is vertices of one tetrehedron
        self.generate_edges()

        # self.encoder, self.in_dim = get_encoder('hashgrid', input_dim=3)
        self.encoder = GridEncoder(input_dim=3, num_levels=16, level_dim=2, base_resolution=16, log2_hashmap_size=19, desired_resolution=2048, gridtype='hash', align_corners=False, interpolation='linear')
        self.in_dim = self.encoder.output_dim
        # self.mlp = MLP(self.in_dim, 4, 32, 3, False)
        if True:
            self.mlp = MLP(self.in_dim, 4, 32, 3, False)
        if False: # CHECK: flip-normal debugging
            self.mlp = MLP(self.in_dim, 4, 32, 3, True)
        self.encoder.cuda()
        self.mlp.cuda()

        self.init_dmtet(self.FLAGS.init_mesh)


    def init_dmtet(self, init_mesh):
        # init sdf from base mesh by open3d
        if init_mesh == "ellipsoid":
            print(f"[INFO] init sdf from open3d sphere")
            sphere = o3d.t.geometry.TriangleMesh.from_legacy(
                o3d.geometry.TriangleMesh.create_sphere(self.FLAGS.ellipsoid_scale))

            min_bound = sphere.vertex.positions.min(0).numpy()  # array([-0.4, -0.4, -0.4], dtype=float32)
            max_bound = sphere.vertex.positions.max(0).numpy()  # array([0.4, 0.4, 0.4], dtype=float32)

            scale = 1.5 / np.array(max_bound - min_bound).max()
            center = np.array(max_bound + min_bound) / 2
            center = center.astype(np.float32)

            sphere.vertex.positions = (sphere.vertex.positions - center) * scale

            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(sphere)

            signed_distance = scene.compute_signed_distance(self.verts.detach().cpu().numpy())
            signed_distance = np.array([s.item() for s in signed_distance]).astype(np.float32)
            signed_distance = torch.from_numpy(signed_distance).cuda()
            signed_distance *= -1  # INNER is POSITIVE
            # min_bound = np.array([-1., -1., -1.], dtype=np.float32)
            # max_bound = np.array([1., 1., 1.], dtype=np.float32)
            
            # pretraining 
            loss_fn = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(list(self.parameters()), lr=1e-3)
            
            # pretrain_iters = 10000
            pretrain_iters = 1000
            batch_size = 10240
            print(f"[INFO] start SDF pre-training ")
            for i in tqdm.tqdm(range(pretrain_iters), desc="[ellipsoid mesh]"):
                # rand_idx = torch.randint(0, self.verts.shape[0], (batch_size,))
                # p = self.verts[rand_idx]
                # ref_value = sdf[rand_idx]
        
                # output = self.mlp(self.encoder(p))
                # loss = loss_fn(output[...,0], ref_value)
                rand_idx = torch.randint(0, self.verts.shape[0], (batch_size,))
                p = self.verts[rand_idx]
                ref_value = signed_distance[rand_idx]

                output = self.mlp(self.encoder(p))
                loss = loss_fn(output[...,0], ref_value)
                torch.nn.utils.clip_grad_norm_(self.parameters(), 5)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # if i % 100 == 0:
                #     print(f"[INFO] SDF pre-train: {loss.item()}")
                    
            print(f"[INFO] SDF pre-train final loss: {loss.item()}")

            # visualize 
            # sdf_np_gt = sdf.cpu().numpy()
            # sdf_np = self.mlp(self.encoder(self.verts)).detach().cpu().numpy()[..., 0]
            # verts_np = self.verts.cpu().numpy()
            # color = np.zeros_like(verts_np)
            # color[sdf_np < 0] = [1, 0, 0]
            # color[sdf_np > 0] = [0, 0, 1]
            # color = (color * 255).astype(np.uint8)
            # pc = trimesh.PointCloud(verts_np, color)
            # axes = trimesh.creation.axis(axis_length=4)
            # box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
            # trimesh.Scene([mesh, pc, axes, box]).show()
        else:
            print(f"[INFO] init sdf from CUSTOM mesh")
            init_shape = get_normalize_mesh(self.FLAGS.init_mesh)
            
            if self.FLAGS.init_mesh_thicker:
                init_shape = get_thicker_mesh(init_shape, self.FLAGS.init_mesh_thicker)

            # if self.FLAGS.init_mesh_type == "rcnn":
            #     init_shape = get_flipped_mesh(init_shape)

            vertices = np.asarray(init_shape.vertices)
            vertices[...,0]=vertices[...,0] * self.FLAGS.sdf_init_shape_scale[0]
            vertices[...,1]=vertices[...,1] * self.FLAGS.sdf_init_shape_scale[1]
            vertices[...,2]=vertices[...,2] * self.FLAGS.sdf_init_shape_scale[2]
            vertices = vertices @ util.rotate_x_2(np.deg2rad(self.FLAGS.sdf_init_shape_rotate_x))
            vertices[...,1]=vertices[...,1] + self.FLAGS.translation_y
            init_shape.vertices = o3d.cuda.pybind.utility.Vector3dVector(vertices)
            # o3d.io.write_triangle_mesh("temp.obj", init_shape)  # DEBUG: mesh save for debugging
            # points_surface = np.asarray(init_shape.sample_points_poisson_disk(5000).points)
            init_shape = o3d.t.geometry.TriangleMesh.from_legacy(init_shape)
            
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(init_shape)

            if True: # CHECK: flip-normal debugging
                ### visualize SDF
                def visualize_sdf(init_shape, scene, name, sign=None):
                    min_bound = init_shape.vertex.positions.min(0).numpy()
                    max_bound = init_shape.vertex.positions.max(0).numpy()
                    xyz_range = np.linspace(min_bound, max_bound, num=32)
                    query_points = np.stack(np.meshgrid(*xyz_range.T), axis=-1).astype(np.float32)
                    _signed_distance = scene.compute_signed_distance(query_points)
                    if sign is not None:
                        _signed_distance *= sign
                    import matplotlib.pyplot as plt
                    plt.clf()
                    plt.imshow(_signed_distance.cpu().numpy()[:, :, 15])
                    plt.colorbar()
                    plt.savefig(f"{name}.png")
                ###

                signed_distance = scene.compute_signed_distance(self.verts.detach().cpu().numpy())
                
                ### visualize
                if False:
                    signed_distance = torch.from_numpy(signed_distance.numpy())
                    rand_idx = torch.randint(0, self.verts.shape[0], (50000,))
                    p = self.verts[rand_idx]
                    sdf_np = signed_distance[rand_idx]

                    verts_np = p.cpu().numpy()
                    color = np.zeros_like(verts_np)
                    color[sdf_np < 0] = [0, 0, 1]  # B
                    color[sdf_np > 0] = [1, 0, 0]  # R
                    color[sdf_np == 0] = [0, 1, 0] # G
                    # color[sdf_np > 0.6] = [1, 0, 0]  # R
                    color = (color * 255).astype(np.uint8)

                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(verts_np)
                    pcd.colors = o3d.utility.Vector3dVector(color)
                    o3d.io.write_point_cloud('x_results/inner_negative.ply', pcd)
                    print("hi")
                ###
                
                # signed_distance = np.array([s.item() for s in signed_distance]).astype(np.float32)
                # signed_distance = torch.from_numpy(signed_distance).cuda()
                signed_distance = torch.from_numpy(signed_distance.numpy()).float().cuda()

                # FIXME: only negative makes larger
                val_min = signed_distance.min()
                val_max = signed_distance.max()
                val_scale = abs(val_min.item()) / abs(val_max.item())
                if val_scale < 1:
                    val_scale = 1 / val_scale
                val_scale *= 5
                
                signed_distance[signed_distance < 0] = signed_distance[signed_distance < 0] * val_scale

                # signed_distance *= -1  # INNER is POSITIVE  # FIXME: disabled to this
                # min_bound = np.array([-1., -1., -1.], dtype=np.float32)
                # max_bound = np.array([1., 1., 1.], dtype=np.float32)
                
                # pretraining 
                loss_fn = torch.nn.MSELoss()
                optimizer = torch.optim.Adam(list(self.parameters()), lr=1e-3)

                # pretrain_iters = 10000
                # TODO: finetuning
                pretrain_iters = 10
                batch_size = 102400
            
            if False:
                signed_distance = scene.compute_signed_distance(self.verts.detach().cpu().numpy())
                signed_distance = np.array([s.item() for s in signed_distance]).astype(np.float32)
                signed_distance = torch.from_numpy(signed_distance).cuda()
                signed_distance *= -1  # INNER is NEGATIVE

                # pretraining 
                loss_fn = torch.nn.MSELoss()
                optimizer = torch.optim.Adam(list(self.parameters()), lr=1e-3)  # FIXME:
                
                # pretrain_iters = 10000
                # TODO: finetuning
                pretrain_iters = 5000
                batch_size = 10240

            print(f"[INFO] start SDF pre-training")
            for i in tqdm.tqdm(range(pretrain_iters),desc="[custom mesh]"):
                # rand_idx = torch.randint(0, self.verts.shape[0], (batch_size,))
                # p = self.verts[rand_idx]
                # ref_value = sdf[rand_idx]
        
                # output = self.mlp(self.encoder(p))
                # loss = loss_fn(output[...,0], ref_value)
                rand_idx = torch.randint(0, self.verts.shape[0], (batch_size,))  # self.verts.shape: torch.Size([277410, 3])
                p = self.verts[rand_idx]
                ref_value = signed_distance[rand_idx]

                output = self.mlp(self.encoder(p))

                ### add noise ### # FIXME:
                # ref_value.mean(): tensor(-0.4910, device='cuda:0')
                # ref_value.std(): tensor(0.2570, device='cuda:0')
                # noise = torch.tensor(np.random.normal(0, 0.1, ref_value.size()), dtype=torch.float32, device='cuda')
                # loss = loss_fn(output[...,0], ref_value + noise)
                ######

                loss = loss_fn(output[...,0], ref_value)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # if i % 100 == 0:
                #     print(f"[INFO] SDF pre-train: {loss.item()}")
                    
            print(f"[INFO] SDF pre-train final loss: {loss.item()}")

            ### visualize
            if False:  # CHECK: flip-normal debugging
                output = self.mlp(self.encoder(self.verts)).detach().cpu().numpy()[..., 0]
                rand_idx = torch.randint(0, self.verts.shape[0], (50000,))
                p = self.verts[rand_idx]
                sdf_np = output[rand_idx]

                verts_np = p.cpu().numpy()
                color = np.zeros_like(verts_np)
                color[sdf_np < 0] = [0, 0, 1]  # B
                color[sdf_np > 0] = [1, 0, 0]  # R
                color[sdf_np == 0] = [0, 1, 0] # G
                color = (color * 255).astype(np.uint8)

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(verts_np)
                pcd.colors = o3d.utility.Vector3dVector(color)
                o3d.io.write_point_cloud('x_results/inner_negative-after_optim.ply', pcd)
                print("hi")
            ###

            # visualize 
            # sdf_np_gt = sdf.cpu().numpy()
            # sdf_np = self.mlp(self.encoder(self.verts)).detach().cpu().numpy()[..., 0]
            # verts_np = self.verts.cpu().numpy()
            # color = np.zeros_like(verts_np)
            # color[sdf_np < 0] = [1, 0, 0]
            # color[sdf_np > 0] = [0, 0, 1]
            # color = (color * 255).astype(np.uint8)
            # pc = trimesh.PointCloud(verts_np, color)
            # axes = trimesh.creation.axis(axis_length=4)
            # box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
            # temp_scene = trimesh.Scene([mesh, pc, axes, box])
            # trimesh.Scene([mesh, pc, axes, box]).show()
       
    def generate_edges(self):
        with torch.no_grad():
            edges = torch.tensor([0,1, 0,2, 0,3, 1,2, 1,3, 2,3], dtype = torch.long, device = "cuda") # six edges for each tetrahedron.
            all_edges = self.indices[:,edges].reshape(-1,2) # [M * 6, 2]; self.indices.shape: torch.Size([1524684, 4])
            all_edges_sorted = torch.sort(all_edges, dim=1)[0]
            self.all_edges = torch.unique(all_edges_sorted, dim=0)

    def getAABB(self):
        return torch.min(self.verts, dim=0).values, torch.max(self.verts, dim=0).values

    def getMesh(self, material):

        pred = self.mlp(self.encoder(self.verts)) # predict SDF and per-vertex deformation
        sdf, deform = pred[:, 0], pred[:, 1:]

        ### visualize
        if False: # CHECK: flip-normal debugging
            rand_idx = torch.randint(0, self.verts.shape[0], (50000,))
            p = self.verts[rand_idx]
            ref_value = sdf[rand_idx]

            sdf_np = ref_value.detach().cpu().numpy()
            verts_np = p.cpu().numpy()
            color = np.zeros_like(verts_np)
            color[sdf_np < 0] = [1, 0, 0]  # R
            color[sdf_np > 0] = [0, 1, 0]  # G
            color[sdf_np == 0] = [0, 0, 1] # B
            color = (color * 255).astype(np.uint8)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(verts_np)
            pcd.colors = o3d.utility.Vector3dVector(color)
            # o3d.io.write_point_cloud('pred_sdf-inv.ply', pcd)
            ###
         
        v_deformed = self.verts + 1 / (self.grid_res) * torch.tanh(deform)

        verts, faces, uvs, uv_idx = self.marching_tets(v_deformed, sdf, self.indices)

        imesh = mesh.Mesh(verts, faces, v_tex=uvs, t_tex_idx=uv_idx, material=material)

        # Run mesh operations to generate tangent space
        imesh = mesh.auto_normals(imesh)
        imesh = mesh.compute_tangents(imesh)  # CHECK: what is compute_tangents?

        return imesh, sdf

    def render(self, glctx, target, lgt, opt_material, bsdf=None, args=None):
        
        # return rendered buffers, keys: ['shaded', 'kd_grad', 'occlusion'].
        opt_mesh, sdf = self.getMesh(opt_material)
        buffers = render.render_mesh(glctx, opt_mesh, target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'], 
                                        msaa=True, background=None, bsdf=bsdf, args=args)
        buffers['mesh'] = opt_mesh
        buffers['sdf'] = sdf

        return buffers


    def tick(self, glctx, target, lgt, opt_material, iteration, is_pixel, args):

        # ==============================================================================================
        #  Render optimizable object with identical conditions
        # ==============================================================================================
        buffers = self.render(glctx, target, lgt, opt_material, args=args)

        # buffers.keys()
        # dict_keys(['shaded', 'kd_grad', 'occlusion', 'normal', 'mesh', 'sdf'])

        mesh = buffers['mesh']

        # ==============================================================================================
        #  Compute loss
        # ==============================================================================================
        t_iter = iteration / self.FLAGS.iter
        
        pred_ws = None
        if self.FLAGS.image:
            # if iteration < int(self.FLAGS.iter * 0.6):
            #     mode = 'normal'
            #     pred_rgb = buffers['normal'][..., 0:3].permute(0, 3, 1, 2).contiguous()
            #     pred_ws = buffers['normal'][..., 3].unsqueeze(1)  # [B, 1, H, W]
            #     pred_rgb = pred_rgb * pred_ws + (1 - pred_ws) * 1  # white bg
            #     as_latent = False
            # else:
            #     mode = 'rgb'
            #     pred_rgb = buffers['shaded'][..., 0:3].permute(0, 3, 1, 2).contiguous()
            #     pred_ws = buffers['shaded'][..., 3].unsqueeze(1) # [B, 1, H, W]
            #     pred_rgb = pred_rgb * pred_ws + (1 - pred_ws) * 1 # white bg
            #     as_latent = False
            # TODO: rgb, normal rendering
            
            # mode = 'rgb'
            # pred_rgb = buffers['shaded'][..., 0:3].permute(0, 3, 1, 2).contiguous()
            # pred_ws = buffers['shaded'][..., 3].unsqueeze(1) # [B, 1, H, W]
            # pred_rgb = pred_rgb * pred_ws + (1 - pred_ws) * 1 # white bg
            # as_latent = False
            
            ###
            if self.FLAGS.geo_normal and (int(self.FLAGS.geo_range[0]) <= iteration) and (iteration <= int(self.FLAGS.geo_range[1])):
                mode = 'normal'
                pred_rgb = buffers['normal'][..., 0:3].permute(0, 3, 1, 2).contiguous()
                pred_ws = buffers['shaded'][..., 3].unsqueeze(1) # [B, 1, H, W]
                pred_rgb = pred_rgb * pred_ws + (1 - pred_ws) * 1 # white bg
                as_latent = False 
            else:
                mode = 'rgb'
                pred_rgb = buffers['shaded'][..., 0:3].permute(0, 3, 1, 2).contiguous()
                pred_ws = buffers['shaded'][..., 3].unsqueeze(1) # [B, 1, H, W]
                pred_rgb = pred_rgb * pred_ws + (1 - pred_ws) * 1 # white bg
                as_latent = False

        else:  # text only
            if iteration < int(self.FLAGS.iter * 0.2):
                mode = 'normal_latent'
                pred_rgb = buffers['normal'][..., 0:4].permute(0, 3, 1, 2).contiguous()
                pred_ws = buffers['normal'][..., 3].unsqueeze(1)  # [B, 1, H, W]
                as_latent = True
            elif iteration < int(self.FLAGS.iter * 0.6):
                mode = 'normal'
                pred_rgb = buffers['normal'][..., 0:3].permute(0, 3, 1, 2).contiguous()
                as_latent = False
            else:
                mode = 'rgb'
                pred_rgb = buffers['shaded'][..., 0:3].permute(0, 3, 1, 2).contiguous()
                pred_ws = buffers['shaded'][..., 3].unsqueeze(1) # [B, 1, H, W]
                pred_rgb = pred_rgb * pred_ws + (1 - pred_ws) * 1 # white bg
                as_latent = False
        
        # FIXME: for material image logging / temporally disabled
        # mat_image = self.FLAGS.mat_interval and (iteration % self.FLAGS.mat_interval == 0)
        # if mat_image and (self.FLAGS.local_rank == 0):
        #     mat_rgb = buffers['shaded'][..., 0:3].permute(0, 3, 1, 2).contiguous()
        #     # initial min, max, mean, std: 0.0, 0.16, 0.05, 0.06
            

        #     mat_ws =  buffers['shaded'][..., 3].unsqueeze(1) # [B, 1, H, W]
            
        #     # mat_rgb = mat_rgb * mat_ws + (1 - mat_ws) * 1  # white bg
        #     mat_rgb = mat_rgb * mat_ws + (1 - mat_ws) * 0  # black bg
        #     mat_rgb = mat_rgb[0].clone().detach().cpu().permute(1, 2, 0).numpy()
        #     mat_rgb = (mat_rgb * 255).round().astype('uint8')

        #     W, H, _ = mat_rgb.shape
        #     mat_rgb = Image.fromarray(mat_rgb)
        #     mat_rgb = mat_rgb.resize((H // 2, W // 2))
        #     mat_rgb.save(os.path.join(self.FLAGS.out_dir, "mat", f"{iteration:07d}.png"))

        # mode = 'rgb'
        # pred_rgb = buffers['shaded'][..., 0:3].permute(0, 3, 1, 2).contiguous()
        # pred_ws = buffers['shaded'][..., 3].unsqueeze(1) # [B, 1, H, W]
        # pred_rgb = pred_rgb * pred_ws + (1 - pred_ws) * 1 # white bg
        # as_latent = False

        args.mode = mode

        def _save_img(img, iteration, path_name):
            img = img.clone().detach().cpu().permute(1, 2, 0).numpy()
            img = (img * 255).round().astype('uint8')

            W, H, _ = img.shape
            img = Image.fromarray(img)
            img = img.resize((H // 2, W // 2))
            img.save(os.path.join(self.FLAGS.out_dir, path_name, f"{iteration:07d}.png"))

        save_image = self.FLAGS.save_interval and (iteration % self.FLAGS.save_interval == 0)
        if (args.wandb_logs) and save_image and (self.FLAGS.local_rank == 0) and (not is_pixel):
            wandb_logs = args.wandb_logs

            # def save_img(img, angle):
            #     img = img.clone().detach().cpu().permute(1, 2, 0).numpy()
            #     img = (img * 255).round().astype('uint8')

            #     W, H, _ = img.shape
            #     img = Image.fromarray(img)
            #     img = img.resize((H // 2, W // 2))
            #     img.save(os.path.join(self.FLAGS.out_dir, "train", f"{angle}.png"))
            # FIXME: save_img function is for debugging
            # save_img(pred_rgb[0], "azimuth_0")
            # save_img(pred_rgb[1], "azimuth_90")
            # save_img(pred_rgb[2], "azimuth_180")
            # save_img(pred_rgb[3], "azimuth_-90")
            
            _pred_rgb = pred_rgb[0].clone().detach().cpu().permute(1, 2, 0).numpy()
            _pred_rgb = (_pred_rgb * 255).round().astype('uint8')

            W, H, _ = _pred_rgb.shape
            _pred_rgb = Image.fromarray(_pred_rgb)
            _pred_rgb = _pred_rgb.resize((H // 2, W // 2))
            _pred_rgb.save(os.path.join(self.FLAGS.out_dir, "train", f"{iteration:07d}.png"))
            wandb_logs.update({'Train/img': wandb.Image(_pred_rgb, caption=f"mode: {mode} / iters: {iteration:06d}"), 'step': iteration})

            if pred_ws != None:
                _pred_ws = pred_ws[0].clone().detach().cpu().permute(1, 2, 0).numpy()
                _pred_ws = (_pred_ws * 255).round().astype('uint8')
                
                _pred_ws = _pred_ws.squeeze()
                W, H = _pred_ws.shape
                _pred_ws = Image.fromarray(_pred_ws)
                _pred_ws = _pred_ws.resize((H // 4, W // 4))
                _pred_ws.save(os.path.join(self.FLAGS.out_dir, "mask", f"{iteration:07d}.png"))
                wandb_logs.update({'Train/mask': wandb.Image(_pred_ws, caption=f"mode: {mode} / iters: {iteration:06d}"), 'step': iteration})
        
        elif is_pixel and save_image and (self.FLAGS.local_rank == 0):
            _save_img(pred_rgb[0], iteration, "pixel")
            
        return buffers, mesh, pred_rgb, pred_ws, self.all_edges, as_latent, iteration, t_iter

