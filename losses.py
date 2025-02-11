import torch
from pytorch3d.ops import knn_points
from pytorch3d.loss import mesh_laplacian_smoothing

# NOTE: Temporary
import ipdb

# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	loss = torch.nn.BCELoss()

	# TODO: Figure out shapes for everything
	ipdb.set_trace()

	return loss(voxel_src, voxel_tgt)

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3  

	# TODO: Figure out shapes for everything
	ipdb.set_trace()

	# Chamfer distance between two point clouds
	src_dist, _ = knn_points(point_cloud_src, point_cloud_tgt)
	tgt_dist, _ = knn_points(point_cloud_tgt, point_cloud_src)

	return torch.mean(src_dist) + torch.mean(tgt_dist)

def smoothness_loss(mesh_src):
	loss_laplacian = mesh_laplacian_smoothing()
	# implement laplacian smoothening loss
	return mesh_laplacian_smoothing(mesh_src)