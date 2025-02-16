import torch
from pytorch3d.ops import knn_points
from pytorch3d.loss import mesh_laplacian_smoothing

# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	loss = torch.nn.BCELoss()

	# IDK WHY - source min is negative sometimes
	voxel_src = torch.clamp(voxel_src, 0, 1)

	return loss(voxel_src, voxel_tgt)

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3  

	# Chamfer distance between two point clouds
	src_dist, _, _ = knn_points(point_cloud_src, point_cloud_tgt)
	tgt_dist, _, _ = knn_points(point_cloud_tgt, point_cloud_src)

	return torch.mean(src_dist) + torch.mean(tgt_dist)

def smoothness_loss(mesh_src):
	# implement laplacian smoothening loss
	return mesh_laplacian_smoothing(mesh_src)