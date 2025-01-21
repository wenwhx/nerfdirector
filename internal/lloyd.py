import os

import numpy as np
import torch
import trimesh as tri
from pytorch3d.ops import knn_points

mesh_path = './utils/mesh/sphere.obj'
mesh = tri.load(mesh_path)
vertices = mesh.vertices
faces = mesh.faces
vertices = np.array(vertices)
faces = np.array(faces).astype(np.int64)
barycenters = (vertices[faces[:, 0]] + vertices[faces[:, 1]] + vertices[faces[:, 2]]) / 3.

def lloyd_relaxation(oldPoint,newPoints,niter=10,top=False,crop=False,mu=np.array([]),mu_mass=np.array([])):
    if mu.shape[0] <= 0 or mu_mass.shape[0] <= 0:
        mu = barycenters / np.linalg.norm(barycenters, axis=1, keepdims=True)
        if top:
            upper_idx = mu[:, 2] >= -0.2
            mu = mu[upper_idx]
        
        if crop:             
            # restrict mu within the range of the cameras            
            all_cameras = np.concatenate([oldPoint, newPoints], axis=0)

            # trim upper bound
            idx = (mu[:, 2] < np.max(all_cameras[:, 2]) + 0.05)
            mu = mu[idx]

            # trim lower bound
            idx = (mu[:, 2] > np.min(all_cameras[:, 2]) - 0.05)
            mu = mu[idx]
        
        mu_mass = np.ones(len(mu)) / len(mu)

    mu_torch = torch.from_numpy(mu).float().cuda()
    mu_mass_torch = torch.from_numpy(mu_mass).float().cuda()

    nb_points = len(newPoints)
    points = np.zeros((len(oldPoint)+len(newPoints),3))
    points[:len(oldPoint)] = oldPoint
    points[len(oldPoint):] = newPoints
    points = torch.from_numpy(points).float().cuda()
    new_points_idx = np.arange(len(oldPoint),len(oldPoint)+nb_points)
    mu_mass_torch_problem = mu_mass_torch.clone()

    for i in range(niter):
        knn = knn_points(mu_torch.unsqueeze(0),points.unsqueeze(0), K=1)
        allocation = knn[1]
        allocation = allocation.squeeze()

        # compute the barycenter 
        bar = torch.zeros_like(points)
        gradient = torch.zeros_like(points)
        for i in new_points_idx:
            mask = allocation == i
            bar[i] = (mu_torch[mask]*mu_mass_torch_problem[mask].unsqueeze(1)).mean(axis=0)
            gradient[i] = bar[i] - points[i]
        
        # Lloyd's update
        points = points + gradient
        points = points/torch.norm(points,dim=1,keepdim=True)
        print('itt {:03d} f : {:1.3e}  ||∇f|| = {:1.3}'.format(i+1,knn[0].mean().item(),gradient.norm(dim=1).mean().item()))
    return points.squeeze().detach().cpu().numpy()
