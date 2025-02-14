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

# NOTE: Temporary
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

def visualize_pointcloud(pointcloud: torch.Tensor, filename: str, args):
    '''
    Render a 360 degree orbit of an pointcloud and save to GIF

    Args:
        pointcloud (torch.Tensor): pointcloud grid to visualize
        filename (str): Output filename to save (e.g. "out.gif")
        args (argparse.Namespace): Object containing args.device
    '''
    feats = torch.ones_like(pointcloud)
    feats = feats * torch.tensor([0.7, 0.7, 1], device=torch.device(args.device))

    cloud = pytorch3d.structures.Pointclouds(
        points=pointcloud,
        features=feats
    )

    orbit(cloud, filename, args)

def visualize_mesh(mesh: pytorch3d.structures.meshes.Meshes, filename: str, args):
    '''
    Render a 360 degree orbit of an mesh and save to GIF

    Args:
        mesh (pytorch3d.structures.meshes.Meshes): mesh grid to visualize
        filename (str): Output filename to save (e.g. "out.gif")
        args (argparse.Namespace): Object containing args.device
    '''
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

def orbit(obj, filename: str, args, inc: int = 10):
    '''
    Render a 360 degree orbit or a mesh and save to GIF

    Args:
        obj: Object to orbit
        filename (string): Output filename to save (e.g. "out.gif")
        args (argparse.Namespace): Object containing args.device
        inc (int): Angle increment in degrees
    '''
    print("Saving %s..." % filename)

    images = []
    for angle in tqdm(range(0, 360, inc)):
        images.append(render(obj, angle, args))

    duration = 1000 // 15
    imageio.mimsave(filename, images, duration=duration, loop=0)

def render(obj, angle: int, args):
    '''
    Render a single view of a mesh

    Args:
        obj: Object to render
        angle (float): Angle to render mesh from
        args (argparse.Namespace): Object containing args.device

    Returns:
        image (np.ndarray): The rendered image
    '''

    R, T = pytorch3d.renderer.look_at_view_transform(3, 0, angle)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=torch.device(args.device))
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=torch.device(args.device))

    if type(obj) == pytorch3d.structures.meshes.Meshes:
        renderer = get_mesh_renderer()
        rend = renderer(obj, cameras=cameras, lights=lights)
    elif type(obj) == pytorch3d.structures.pointclouds.Pointclouds:
        renderer = get_points_renderer()
        rend = renderer(obj, cameras=cameras)
    else:
        raise(NotImplementedError)

    rend = (rend.cpu().detach().numpy()[0, ..., :3] * 255).astype(np.uint8)

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

def get_points_renderer(
    image_size=512, device=None, radius=0.01, background_color=(1, 1, 1)
):
    """
    Returns a Pytorch3D renderer for point clouds.

    Args:
        image_size (int): The rendered image size.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
        radius (float): The radius of the rendered point in NDC.
        background_color (tuple): The background color of the rendered image.
    
    Returns:
        PointsRenderer.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer