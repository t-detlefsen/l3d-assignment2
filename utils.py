import torch
import imageio
import pytorch3d
from tqdm import tqdm
import numpy as np

from pytorch3d.renderer import (
    AlphaCompositor,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    HardPhongShader,
)

import ipdb

def visualize_voxel(voxels: torch.Tensor, filename: str, args):
    '''
    Render a 360 degree orbit of an voxel grid and save to GIF

    Args:
        voxels (torch.Tensor): Voxel grid to visualize
        filename (str): Output filename to save (e.g. "out.gif")
        args (argparse.Namespace): Object containing args.device
    '''
    mesh = pytorch3d.ops.cubify(voxels, 0.5, device=torch.device(args.device))

    vertices = mesh.verts_list()[0].unsqueeze(0)
    faces = mesh.faces_list()[0].unsqueeze(0)
    textures = torch.ones_like(torch.tensor(vertices))
    textures = textures * torch.tensor([0.7, 0.7, 1], device=torch.device(args.device))
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures)
    )
    mesh = mesh.to(torch.device(args.device))

    orbit(mesh, filename, args)

def orbit(mesh: pytorch3d.structures.meshes.Meshes, filename: str, args, inc: int = 10):
    '''
    Render a 360 degree orbit or a mesh and save to GIF

    Args:
        mesh (pytorch3d.structures.meshes.Meshes): Mesh object to orbit
        filename (string): Output filename to save (e.g. "out.gif")
        args (argparse.Namespace): Object containing args.device
        inc (int): Angle increment in degrees
    '''
    images = []
    for angle in tqdm(range(0, 360, inc)):
        images.append(render(mesh, angle, args))

    duration = 1000 // 15
    imageio.mimsave(filename, images, duration=duration, loop=0)

def render(mesh: pytorch3d.structures.meshes.Meshes, angle: int, args):
    '''
    Render a single view of a mesh

    Args:
        mesh (pytorch3d.structures.meshes.Meshes): Mesh object to render
        angle (float): Angle to render mesh from
        args (argparse.Namespace): Object containing args.device

    Returns:
        image (np.ndarray): The rendered image
    '''
    renderer = get_mesh_renderer()

    R, T = pytorch3d.renderer.look_at_view_transform(3, 0, angle)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=torch.device(args.device))
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=torch.device(args.device))

    rend = renderer(mesh, cameras=cameras, lights=lights)
    rend = (rend.cpu().numpy()[0, ..., :3] * 255).astype(np.uint8)

    return rend

def get_mesh_renderer(image_size=512, lights=None, device=None):
    """
    Returns a Pytorch3D Mesh Renderer.

    Args:
        image_size (int): The rendered image size.
        lights: A default Pytorch3D lights object.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(device=device, lights=lights),
    )
    return renderer