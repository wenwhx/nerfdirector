import numpy as np
import trimesh as tri

from internal.lloyd import lloyd_relaxation


def vmf_pdf(mu, kappa, x):
	
	mu = mu.reshape(1, 3)
	x = x.reshape(-1, 3)
	
	# numerically stable implementation from https://people.csail.mit.edu/jstraub/download/straub2017vonMisesFisherInference.pdf
	pdf = kappa * np.exp(kappa * ((mu @ x.T).squeeze() - 1.0))
	pdf = pdf / (2*np.pi*(1.0-np.exp(-2*kappa)))
	
	return pdf


def mvmf_pdf(mus, kappas, alphas, x):
	mus = mus.reshape(-1, 3)
	x = x.reshape(-1, 3)

	pdf = np.zeros((alphas.shape[0], x.shape[0]))
	for k in range(alphas.shape[0]):
		pdf[k] = vmf_pdf(mus[k], kappas[k], x)

	pdf = alphas.reshape(1, -1) @ pdf 

	return pdf.ravel()


def vmf_sample(mu, kappa, N):

	# uniform samply on the unit circle 	
	v = 2 * np.pi * np.random.random(N)
	nx = np.cos(v); ny = np.sin(v);

	# sample u according to F
	u = np.random.random(N)
	u = 1.0 + (1.0/kappa) * np.log(u + (1.0 - u)*np.exp(-2*kappa)) 

	# create samples in m direction
	temp = np.sqrt(1. - np.square(u))
	n = np.column_stack([temp*nx, temp*ny, u])

	# rotate samples to mu direction
	m = np.array([0.0, 0.0, 1.0]); axis_w = np.cross(m, mu); w1, w2, w3 = axis_w;
	s = axis_w.dot(axis_w)**0.5; c= np.dot(m, mu);
	R = None
	if s == 0: 
		R = np.eye(3)
	else:
		wx = np.array([[0, -w3, w2], [w3, 0, -w1], [-w2, w1, 0]]); wx2 = wx @ wx;
		R = np.eye(3) + wx + wx2 * (1-c)/s**2
	
	return (R @ n.T).T


def mvmf_sample(mus, kappas, alphas, N):

	samples = []	
	num_samples = np.random.multinomial(N, pvals=alphas / alphas.sum(), size=1).ravel()
	for i, n in enumerate(num_samples):		
		samples.append(vmf_sample(mus[i], kappa=float(kappas[i]), N=n))
	samples = np.concatenate(samples, axis=0)
	return samples


def vMF_sampling(K, eval_values, cur_centers, val_centers, sigma=5.0, kappa=50.0, top=False, lloyd=True, crop=False):
	''' von Mises-Fisher sampling
	
	Args:
	K: int, the number of sample points
	eval_values: list, the list storing the eval metric values for val_centers
	cur_centers: list, the list of current training centers
	val_centers: list, the list of training candidates (sorted by their name)
	sigma: float
	kappa: float
	top: bool
	lloyd: bool, indicator of enabling lloyd relaxation
	crop: bool, indicator of adjusting the barycenters range
	'''
	lloyd_iter = 10 if not crop or (crop and cur_centers.shape[0] < 60) else 3
	print('SIGMA:{} KAPPA:{} lloyd_iter:{} crop:{}\n'.format(sigma, kappa, lloyd_iter, crop))
	eval_values = np.array(eval_values)
	
	# normalized current center
	norm_cur_centers = cur_centers / np.linalg.norm(cur_centers, axis=1, keepdims=True)

	# mvMF parameters
	mus = val_centers / np.linalg.norm(val_centers, axis=1, keepdims=True)
	kappas = np.ones(mus.shape[0]) * kappa
	alphas = np.exp((eval_values.max() - eval_values)/(eval_values.max()-eval_values.min())/ sigma)
	alphas = alphas / np.sum(alphas)

	# sample points
	sampled_views = mvmf_sample(mus, kappas, alphas, K)

	if lloyd:
		mesh_path = './utils/mesh/sphere.obj'
		mesh = tri.load(mesh_path)
		vertices = mesh.vertices
		faces = mesh.faces
		vertices = np.array(vertices)
		faces = np.array(faces).astype(np.int64)
		barycenters = (vertices[faces[:, 0]] + vertices[faces[:, 1]] + vertices[faces[:, 2]]) / 3.
		mu = barycenters / np.linalg.norm(barycenters, axis=1, keepdims=True)

		if top:
			upper_idx = mu[:, 2] >= -0.1
			mu = mu[upper_idx]

		if crop:

			# collect all candidates
			all_cameras = np.concatenate([norm_cur_centers, mus], axis=0)

			# trim upper bound
			idx = (mu[:, 2] < np.max(all_cameras[:, 2]) + 0.05)
			mu = mu[idx]

			# trim lower bound
			idx = (mu[:, 2] > np.min(all_cameras[:, 2]) - 0.05)
			mu = mu[idx]

		mu_mass = mvmf_pdf(mus, kappas, alphas, mu)
		mu_mass = mu_mass / mu_mass.sum()

		new_points = lloyd_relaxation(norm_cur_centers, sampled_views, niter=lloyd_iter, top=top, crop=False, mu=mu, mu_mass=mu_mass)
	else:
		new_points = sampled_views

	new_points = new_points[-K:]
	return new_points
