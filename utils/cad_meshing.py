import logging
import numpy as np
import skimage.measure
import time
import torch
import utils.utils
import os
import mcubes
import pymesh
from torch.autograd import Variable
import trimesh
import math

#marching cube to get estimated shape surface
def create_mesh_mc(
	generator, phase, primitives, filename, N=128, max_batch=32 ** 3, threshold=0.5
):
	start = time.time()
	ply_filename = filename
	# NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
	voxel_origin = [0, 0, 0]
	voxel_size = 1

	overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
	samples = torch.zeros(N ** 3, 4) # x y z sdf cell_num
	# p *5 
	# transform first 3 columns
	# to be the x, y, z index
	samples[:, 2] = overall_index % N
	samples[:, 1] = (overall_index.long() / N) % N
	samples[:, 0] = ((overall_index.long() / N) / N) % N

	samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
	samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
	samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

	samples[:, :3] = (samples[:, :3]+0.5)/N-0.5

	num_samples = N ** 3

	samples.requires_grad = False

	head = 0
	
	while head < num_samples:
		sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()

		_, _, occ, _ = generator(sample_subset.unsqueeze(0), primitives, phase)
		samples[head : min(head + max_batch, num_samples), 3] = (
			occ.reshape(-1)
			.detach()
			.cpu()
		)
		head += max_batch

	sdf_values = samples[:, 3]
	sdf_values = sdf_values.reshape(N, N, N)
			
	end = time.time()
	print("sampling takes: %f" % (end - start))

	numpy_3d_sdf_tensor = sdf_values.numpy()

	verts, faces = mcubes.marching_cubes(numpy_3d_sdf_tensor, threshold)

	mesh_points = verts
	
	mesh_points = (mesh_points+0.5)/N-0.5
	
	if not os.path.exists(os.path.dirname(ply_filename)):
		os.makedirs(os.path.dirname(ply_filename))

	utils.utils.write_ply_triangle(ply_filename + ".ply", mesh_points, faces)

#get point inside/outside each convex indicator
def point_convex_estimator(generator, samples, primitives, c_dim, max_batch=int(2 ** 14)):
	#input points, primitives
	#output, point-convex indicator
	samples = torch.from_numpy(samples)
	primitives = torch.from_numpy(primitives)
	samples.requires_grad = False
	model_float = torch.ones(samples.shape[0], c_dim)
	num_samples = samples.shape[0]
	head = 0
	while head < num_samples:
		sample_subset = samples[head : min(head + max_batch, num_samples), :].cuda()
		_, _, _, h2 = generator(sample_subset.unsqueeze(0), primitives.cuda(), phase=2)
		model_float[head : min(head + max_batch, num_samples), :] = h2.reshape(-1, c_dim).detach().cpu()
		head += max_batch
	return model_float.numpy()

#sample points on estimated shape surface
def sample_points_around_estimated_surface(generator, samples, primitives, N, c_dim, max_batch=int(2 ** 14), threshold=0.9):
	
	samples = torch.from_numpy(samples)
	primitives = torch.from_numpy(primitives)
	samples.requires_grad = False
	model_float = torch.ones(N ** 3)
	num_samples = N ** 3
	head = 0
	while head < num_samples:
		sample_subset = samples[head : min(head + max_batch, num_samples), :].cuda()
		_, _, h3, _ = generator(sample_subset.unsqueeze(0), primitives.cuda(), phase=2)
		model_float[head : min(head + max_batch, num_samples)] = h3.reshape(-1).detach().cpu()
		head += max_batch

	sdf_values = model_float
	sdf_values = sdf_values.reshape(N, N, N)

	verts, faces = mcubes.marching_cubes(sdf_values.numpy(), threshold)

	mesh_points = verts
	
	mesh_points = (mesh_points+0.5)/N-0.5

	mesh = trimesh.Trimesh(mesh_points, faces)
	points, _ = mesh.sample(50000, return_index=True)
	sigma = 1/64
	points = points + sigma * (np.random.random_sample((points.shape[0], points.shape[1]))*2 - 1)
	return points.astype(np.float32)

#
def mask_model_float(model_float, convex_mask):
	model_float = np.expand_dims(model_float, axis=0)
	model_float = model_float * (1 - convex_mask)
	return model_float

#meshing each primitive
def get_implicit_field_for_one_primitive(points, primitives):
	primitives = torch.from_numpy(primitives)
	#points = torch.from_numpy(points)
	points = torch.from_numpy(points).float().requires_grad_(True)
	
	ones = torch.ones((points.shape[0], points.shape[1], 1))
	pointsx = torch.cat([(points)**2, points, ones], 2)
	h1 = torch.matmul(pointsx, primitives)
	return h1.detach().numpy()

def get_grid_samples(N):
	voxel_origin = [0, 0, 0]
	overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
	voxel_size = 1

	samples = torch.ones(N ** 3, 3)

	# transform first 3 columns
	# to be the x, y, z index
	samples[:, 2] = overall_index % N
	samples[:, 1] = (overall_index.long() / N) % N
	samples[:, 0] = ((overall_index.long() / N) / N) % N

	# transform first 3 columns
	# to be the x, y, z coordinate
	samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
	samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
	samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

	samples = (samples+0.5)/N-0.5

	return samples.numpy()

#meshing all primitives, convex, and final shape
def create_cad_mesh(generator, primitives, connection_t, filename, file_type='off', p_dim=1024, \
	c_dim=64, N=128, max_batch=32 ** 3, shape_repair_flag=False, inter_results_flag=False, final_result_flag=False, primi_count_only_flag=False):
	start_time = time.time()
	ply_filename = filename
	connection_t = connection_t.unsqueeze(0)
	primitives = primitives.detach().cpu().numpy()

	if not os.path.isdir(ply_filename):
		os.makedirs(ply_filename)
	
	mc_res = int(N/2)
	sampling_threshold = 0.9
	samples_grid_high = get_grid_samples(N)
	samples_surface = sample_points_around_estimated_surface(generator, samples_grid_high, 
		primitives, N, c_dim, threshold = sampling_threshold)
	samples_grid_low = get_grid_samples(mc_res)

	original_primitives = np.copy(primitives)

	#coefficients regularization
	primitives = primitives*( (np.sum(np.abs(primitives[:,0:3]), axis=1, keepdims=True)>1e-3) | (np.sum(np.abs(primitives[:,3:6]), axis=1, keepdims=True)>1e-3) ).astype(np.float32)
	primitives = primitives/np.maximum(np.sqrt(np.sum(np.square(primitives[:,0:6]), axis=1, keepdims=True)),1e-6)
	primitives = primitives*1e4

	#point-convex inside/outside estimator
	#P, C
	model_float = point_convex_estimator(generator, samples_surface, primitives, c_dim)
	#inside 1, outside 0
	model_float = (model_float > sampling_threshold).astype(np.uint8)
	model_float_copy = np.copy(model_float)
	c_dim_half = c_dim//2
	model_float_sum = np.minimum(np.max(model_float[:,:c_dim_half],axis=1),1-np.max(model_float[:,c_dim_half:],axis=1))

	unused_convex = np.ones([c_dim], np.uint8)

	for i in range(c_dim):
		if np.max(model_float[:,i])>0:
			model_float_copy[:,i] = 0
			#whether keep this convex
			model_float_sum_new = np.minimum(np.max(model_float_copy[:,:c_dim_half],axis=1),1-np.max(model_float_copy[:,c_dim_half:],axis=1))
			varied_output = (model_float_sum_new^model_float_sum).squeeze().astype(int)
			changed_point_num = np.sum(varied_output)
			if changed_point_num > 1:
				#print(f'convex id {i}, change num {changed_point_num}')
				model_float_copy[:,i] = model_float[:,i]
				unused_convex[i] = 0
			else:
				model_float[:,i] = 0
				model_float_sum = np.logical_or(model_float_sum, model_float_sum_new)

	convex_num = c_dim - np.sum(unused_convex)
	print('convex num, ', convex_num)

	convex_mask = np.reshape(unused_convex, [1,1,-1])
	
	for i in range(c_dim):
		if unused_convex[i]:
			connection_t[0,:,i] = 0

	print('remove convex time:, ', time.time() - start_time)

	#start checking primitives for each convex
	zg = point_convex_estimator(generator, samples_surface, primitives, c_dim)
	correct_output = mask_model_float(zg, convex_mask)
	correct_output = correct_output > sampling_threshold

	connection_t_temp = connection_t>0.01
	connection_t_temp = connection_t_temp.squeeze()
	connection_t_temp = np.sum(connection_t_temp.detach().cpu().numpy(), axis=1)
	connection_t_temp = np.argwhere(connection_t_temp > 0)
	primitives_temp = np.copy(primitives)
	print('primi needs to be checked, ', len(connection_t_temp))
	primi_count = 0
	for i in connection_t_temp:
		primitives_temp[0,:,i] = 0
		zg = point_convex_estimator(generator, samples_surface, primitives_temp, c_dim)
		this_output = mask_model_float(zg, convex_mask)
		this_output = this_output > sampling_threshold
		varied_output = (correct_output^this_output).squeeze().astype(int)
		changed_point_num = np.sum(varied_output)
		if changed_point_num > 1:
			primitives_temp[0,:,i] = primitives[0,:,i]
			primi_count += 1
		else:
			primitives[0,:,i] = 0
			correct_output = np.logical_or(correct_output, this_output)
			
	print('remove plane time:, ', time.time() - start_time)
	print('remained primi count:, ', primi_count)
	connection_t = (connection_t>0.01).float()
	for i in range(p_dim):
		if np.all(primitives[0,:,i]==0):
			connection_t[0,i,:] = 0

	#csg operations
	print('csg start time:, ', time.time() - start_time)
	dimf = mc_res
	model_float = np.full([dimf+2,dimf+2,dimf+2],100,np.float32)
	output_mesh = None
	concave_mesh_left = None
	concave_mesh_right = None
	primi_dir = ply_filename + '/primi_for_each_convex'
	convex_dir = ply_filename + '/convex'
	concave_dir = ply_filename + '/concave'

	if not os.path.isdir(primi_dir):
		os.makedirs(primi_dir)
	if not os.path.isdir(convex_dir):
		os.makedirs(convex_dir)
	if not os.path.isdir(concave_dir):
		os.makedirs(concave_dir)

	for i in range(c_dim):
		if unused_convex[i]==0:
			print('convex %d, time: %f'%(i, time.time() - start_time))
			convex_mesh = None
			for j in range(p_dim):
				if connection_t[0,j,i]>0:
					model_out = get_implicit_field_for_one_primitive(np.expand_dims(samples_grid_low, axis=0), original_primitives[:,:,j:j+1])
					#bounded by unit box, solid for CSG
					model_float[1:-1,1:-1,1:-1] = np.reshape(model_out, [dimf,dimf,dimf])
					if i<c_dim_half:
						vertices, triangles = mcubes.marching_cubes(-model_float, 0)
					else:
						vertices, triangles = mcubes.marching_cubes(-model_float, -0.05)

					vertices = (vertices-0.5)/dimf-0.5

					if convex_mesh is None:
						convex_mesh = pymesh.form_mesh(vertices, triangles)
					else:
						convex_mesh = pymesh.boolean(convex_mesh, pymesh.form_mesh(vertices, triangles), operation="intersection")

					#open surface mesh, no bound
					model_out = np.reshape(model_out, [dimf,dimf,dimf])
					if i<c_dim_half:
						vertices, triangles = mcubes.marching_cubes(-model_out, 0)
					else:
						vertices, triangles = mcubes.marching_cubes(-model_out, -0)

					vertices = (vertices+0.5)/dimf-0.5
					if inter_results_flag:
						pymesh.meshio.save_mesh_raw(primi_dir +"/"+str(i)+ "_"+ str(j) +"." + file_type, vertices, triangles)
					
			if inter_results_flag:		
				pymesh.meshio.save_mesh_raw(convex_dir +"/"+str(i)+"." + file_type, convex_mesh.vertices, convex_mesh.faces)
			
			if i < c_dim_half:
				if concave_mesh_left is None:
					concave_mesh_left = convex_mesh
				else:
					concave_mesh_left = pymesh.boolean(concave_mesh_left, convex_mesh, operation="union")
			else:
				if concave_mesh_right is None:
					concave_mesh_right = convex_mesh
				else:
					concave_mesh_right = pymesh.boolean(concave_mesh_right, convex_mesh, operation="union")	
	if concave_mesh_left is None:
		if concave_mesh_right is None:
			print('empty mesh!!!')
			exit()
		else:
			output_mesh = concave_mesh_right
			if inter_results_flag:
				pymesh.meshio.save_mesh_raw(concave_dir +"/right"+"." + file_type, concave_mesh_right.vertices, concave_mesh_right.faces)
	else:
		if concave_mesh_right is None:
			output_mesh = concave_mesh_left
			if inter_results_flag:
				pymesh.meshio.save_mesh_raw(concave_dir +"/left"+"." + file_type, concave_mesh_left.vertices, concave_mesh_left.faces)
		else:
			output_mesh = pymesh.boolean(concave_mesh_left, concave_mesh_right, operation="difference")
			if inter_results_flag:
				pymesh.meshio.save_mesh_raw(concave_dir +"/left"+"." + file_type, concave_mesh_left.vertices, concave_mesh_left.faces)
				pymesh.meshio.save_mesh_raw(concave_dir +"/right"+"." + file_type, concave_mesh_right.vertices, concave_mesh_right.faces)

	if final_result_flag:
		file_name = os.path.basename(ply_filename)
		pymesh.meshio.save_mesh_raw(os.path.join(ply_filename, file_name + "." + file_type), output_mesh.vertices, output_mesh.faces)

	return convex_num, primi_count