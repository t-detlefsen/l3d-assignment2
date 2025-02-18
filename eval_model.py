import argparse
import time
import torch
from model import SingleViewto3D
from r2n2_custom import R2N2
from  pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
import dataset_location
import pytorch3d
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.ops import knn_points
import mcubes
import utils_vox
import matplotlib.pyplot as plt 
from pytorch3d.transforms import Rotate, axis_angle_to_matrix
import math
import numpy as np

from utils import *

def get_args_parser():
    parser = argparse.ArgumentParser('Singleto3D', add_help=False)
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--vis_freq', default=1000, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh'], type=str)
    parser.add_argument('--n_points', default=1000, type=int)
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=0.1, type=float)  
    parser.add_argument('--load_checkpoint', action='store_true')  
    parser.add_argument('--device', default='cuda', type=str) 
    parser.add_argument('--load_feat', action='store_true')
    parser.add_argument('--vis_all',  action='store_true') 
    return parser

def preprocess(feed_dict, args):
    for k in ['images']:
        feed_dict[k] = feed_dict[k].to(args.device)

    images = feed_dict['images'].squeeze(1)
    mesh = feed_dict['mesh']
    if args.load_feat:
        images = torch.stack(feed_dict['feats']).to(args.device)

    return images, mesh

def save_plot(thresholds, avg_f1_score, args):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(thresholds, avg_f1_score, marker='o')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1-score')
    ax.set_title(f'Evaluation {args.type}')
    plt.savefig(f'eval_{args.type}', bbox_inches='tight')


def compute_sampling_metrics(pred_points, gt_points, thresholds, eps=1e-8):
    metrics = {}
    lengths_pred = torch.full(
        (pred_points.shape[0],), pred_points.shape[1], dtype=torch.int64, device=pred_points.device
    )
    lengths_gt = torch.full(
        (gt_points.shape[0],), gt_points.shape[1], dtype=torch.int64, device=gt_points.device
    )

    # For each predicted point, find its neareast-neighbor GT point
    knn_pred = knn_points(pred_points, gt_points, lengths1=lengths_pred, lengths2=lengths_gt, K=1)
    # Compute L1 and L2 distances between each pred point and its nearest GT
    pred_to_gt_dists2 = knn_pred.dists[..., 0]  # (N, S)
    pred_to_gt_dists = pred_to_gt_dists2.sqrt()  # (N, S)

    # For each GT point, find its nearest-neighbor predicted point
    knn_gt = knn_points(gt_points, pred_points, lengths1=lengths_gt, lengths2=lengths_pred, K=1)
    # Compute L1 and L2 dists between each GT point and its nearest pred point
    gt_to_pred_dists2 = knn_gt.dists[..., 0]  # (N, S)
    gt_to_pred_dists = gt_to_pred_dists2.sqrt()  # (N, S)

    # Compute precision, recall, and F1 based on L2 distances
    for t in thresholds:
        precision = 100.0 * (pred_to_gt_dists < t).float().mean(dim=1)
        recall = 100.0 * (gt_to_pred_dists < t).float().mean(dim=1)
        f1 = (2.0 * precision * recall) / (precision + recall + eps)
        metrics["Precision@%f" % t] = precision
        metrics["Recall@%f" % t] = recall
        metrics["F1@%f" % t] = f1

    # Move all metrics to CPU
    metrics = {k: v.cpu() for k, v in metrics.items()}
    return metrics

def evaluate(predictions, mesh_gt, thresholds, args):
    if args.type == "vox":
        voxels_src = predictions
        H,W,D = voxels_src.shape[2:]
        vertices_src, faces_src = mcubes.marching_cubes(voxels_src.detach().cpu().squeeze().numpy(), isovalue=0.5)
        vertices_src = torch.tensor(vertices_src).float()
        faces_src = torch.tensor(faces_src.astype(int))
        mesh_src = pytorch3d.structures.Meshes([vertices_src], [faces_src])
        pred_points = sample_points_from_meshes(mesh_src, args.n_points)
        pred_points = utils_vox.Mem2Ref(pred_points, H, W, D)
        # Apply a rotation transform to align predicted voxels to gt mesh
        angle = -math.pi
        axis_angle = torch.as_tensor(np.array([[0.0, angle, 0.0]]))
        Rot = axis_angle_to_matrix(axis_angle)
        T_transform = Rotate(Rot)
        pred_points = T_transform.transform_points(pred_points)
        # re-center the predicted points
        pred_points = pred_points - pred_points.mean(1, keepdim=True)
    elif args.type == "point":
        pred_points = predictions.cpu()
    elif args.type == "mesh":
        pred_points = sample_points_from_meshes(predictions, args.n_points).cpu()

    gt_points = sample_points_from_meshes(mesh_gt, args.n_points)
    if args.type == "vox":
        gt_points = gt_points - gt_points.mean(1, keepdim=True)
    metrics = compute_sampling_metrics(pred_points, gt_points, thresholds)
    return metrics



def evaluate_model(args):
    r2n2_dataset = R2N2("test", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True, return_feats=args.load_feat)

    loader = torch.utils.data.DataLoader(
        r2n2_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_batched_R2N2,
        pin_memory=True,
        drop_last=True)
    eval_loader = iter(loader)

    model = SingleViewto3D(args)
    model.to(args.device)
    model.eval()

    start_iter = 0
    start_time = time.time()

    thresholds = [0.01, 0.02, 0.03, 0.04, 0.05]

    avg_f1_score_05 = []
    avg_f1_score = []
    avg_p_score = []
    avg_r_score = []

    if args.load_checkpoint:
        checkpoint = torch.load(f'checkpoint_{args.type}.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Succesfully loaded iter {start_iter}")
    
    print("Starting evaluating !")
    max_iter = len(eval_loader)
    for step in range(start_iter, max_iter):
        iter_start_time = time.time()

        read_start_time = time.time()

        feed_dict = next(eval_loader)

        images_gt, mesh_gt = preprocess(feed_dict, args)

        read_time = time.time() - read_start_time

        predictions = model(images_gt, args)

        try:
            metrics = evaluate(predictions, mesh_gt, thresholds, args)
        except:
            print("Empty Mesh, skipping")
            continue

        if (step % args.vis_freq) == 0:
            # Process Ground Truth
            vertices = mesh_gt.verts_list()[0].unsqueeze(0)
            faces = mesh_gt.faces_list()[0].unsqueeze(0)
            textures = torch.ones_like(torch.tensor(vertices))
            textures = textures * torch.tensor([0.7, 0.7, 1])
            new_mesh_gt = pytorch3d.structures.Meshes(
                verts=vertices,
                faces=faces,
                textures=pytorch3d.renderer.TexturesVertex(textures)
            )
            new_mesh_gt = new_mesh_gt.to(torch.device(args.device))

            # Process outputs
            if args.type == 'vox':
                if args.vis_all:
                    thresh = [0.45, 0.475, 0.5, 0.525, 0.55, 1.0]
                    textures = torch.zeros((1, 1, 3), device=torch.device(args.device))
                    mesh_list = []
                    for i in range(len(thresh) - 1):
                        pred = torch.where(predictions > thresh[i], predictions, torch.tensor(0))
                        pred = torch.where(pred < thresh[i+1], pred, torch.tensor(0))
                        if torch.sum(pred) == 0:
                            continue
                        mesh = pytorch3d.ops.cubify(pred.squeeze(0), thresh[i], device=torch.device(args.device))
                        vertices = mesh.verts_list()[0].unsqueeze(0)
                        faces = mesh.faces_list()[0].unsqueeze(0)
                        color = torch.tensor([1, 0, 0]) * (1 - i/(len(thresh) - 2)) + torch.tensor([0, 1, 0]) * (i / (len(thresh) - 2))
                        textures = torch.ones_like(mesh.verts_list()[0].unsqueeze(0)) * color.to(torch.device(args.device))
                        pred = pytorch3d.structures.Meshes(
                            verts=vertices,
                            faces=faces,
                            textures=pytorch3d.renderer.TexturesVertex(textures)
                        )
                        mesh_list.append(pred)

                    pred = pytorch3d.structures.join_meshes_as_scene(mesh_list, True)
                    pred = pred.to(torch.device(args.device))

                    orbit(pred, f'outputs/{step}_{args.type}_pred.gif', args)
                    exit()
                else:
                    mesh = pytorch3d.ops.cubify(predictions.squeeze(0), 0.5, device=torch.device(args.device))
                    vertices = mesh.verts_list()[0].unsqueeze(0)
                    faces = mesh.faces_list()[0].unsqueeze(0)
                    textures = torch.ones_like(torch.tensor(vertices))
                    textures = textures * torch.tensor([0.7, 0.7, 1], device=torch.device(args.device))
                    pred = pytorch3d.structures.Meshes(
                        verts=vertices,
                        faces=faces,
                        textures=pytorch3d.renderer.TexturesVertex(textures)
                    )
                    pred = pred.to(torch.device(args.device))
            elif args.type == 'point':
                feats = torch.ones_like(predictions)
                feats = feats * torch.tensor([0.7, 0.7, 1], device=torch.device(args.device))
                pred = pytorch3d.structures.Pointclouds(
                    points=predictions,
                    features=feats
                )
            elif args.type == 'mesh':
                vertices = predictions.verts_list()[0].unsqueeze(0)
                faces = predictions.faces_list()[0].unsqueeze(0)
                textures = torch.ones_like(torch.tensor(vertices))
                textures = textures * torch.tensor([0.7, 0.7, 1], device=torch.device(args.device))
                pred = pytorch3d.structures.Meshes(
                    verts=vertices,
                    faces=faces,
                    textures=pytorch3d.renderer.TexturesVertex(textures)
                )
                pred = pred.to(torch.device(args.device))

            # Save input RGB
            plt.imsave(f'outputs/{step}_{args.type}_rgb.png', (images_gt.squeeze(0).cpu().numpy() * 255).astype(np.uint8))
            # Save Ground Truth
            plt.imsave(f'outputs/{step}_{args.type}_gt.png', render(new_mesh_gt, args))
            # Save Prediction
            if not args.vis_all:
                plt.imsave(f'outputs/{step}_{args.type}_pred.png', render(pred, args))

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        f1_05 = metrics['F1@0.050000']
        avg_f1_score_05.append(f1_05)
        avg_p_score.append(torch.tensor([metrics["Precision@%f" % t] for t in thresholds]))
        avg_r_score.append(torch.tensor([metrics["Recall@%f" % t] for t in thresholds]))
        avg_f1_score.append(torch.tensor([metrics["F1@%f" % t] for t in thresholds]))

        print("[%4d/%4d]; ttime: %.0f (%.2f, %.2f); F1@0.05: %.3f; Avg F1@0.05: %.3f" % (step, max_iter, total_time, read_time, iter_time, f1_05, torch.tensor(avg_f1_score_05).mean()))
    

    avg_f1_score = torch.stack(avg_f1_score).mean(0)

    save_plot(thresholds, avg_f1_score,  args)
    print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Singleto3D', parents=[get_args_parser()])
    args = parser.parse_args()
    evaluate_model(args)
