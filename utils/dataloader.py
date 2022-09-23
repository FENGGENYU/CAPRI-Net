import glob
import logging
import numpy as np
import os
import random
import torch
import torch.utils.data
import h5py
import utils.workspace as ws

class SurfaceSamples(torch.utils.data.Dataset):
	def __init__(
		self,
		data_source,
		test_flag
	):
		print('data source', data_source)
		print('class SurfaceSamples for point cloud reconstruction')
		self.data_source = data_source

		name_file = os.path.join(self.data_source, 'test_names.npz')
		npz_shapes = np.load(name_file)

		if test_flag:
			filename_shapes = os.path.join(self.data_source, 'points2mesh.hdf5')
			data_names = npz_shapes['test_names']
		else:
			filename_shapes = os.path.join(self.data_source, 'points2mesh.hdf5')
			data_names = npz_shapes['test_names']

		data_dict = h5py.File(filename_shapes, 'r')
		data_points = torch.from_numpy(data_dict['points'][:]).float()
		data_voxels = torch.from_numpy(data_dict['voxels'][:]).float()
		data_voxels = data_voxels.squeeze(-1).unsqueeze(1)

		print('loaded points shape totally,', data_points.shape)

		sigma_64 = 1/64
		surface_points = data_points[:, :, :3]
		surface_normals = data_points[:, :, 3:]
		
		repeat = 8

		noise_64 = torch.rand(surface_points.shape[0], repeat*surface_points.shape[1])*2 - 1
		values_64 = (noise_64 <= 0).float()
		noise_64 = noise_64.unsqueeze(2).repeat(1, 1, 3)
		surface_points_repeat = surface_points.repeat(1, repeat, 1)
		surface_normals_repeat = surface_normals.repeat(1, repeat, 1)

		sample_points_64 = surface_points_repeat + sigma_64 * surface_normals_repeat * noise_64
		
		print('surface points sample shape, ', sample_points_64.shape)

		self.data_points = torch.cat([sample_points_64, values_64.unsqueeze(2)], 2)
		self.data_voxels = data_voxels
		self.data_names = data_names
		logging.debug(
			"using "
			+ str(len(self.data_names))
			+ " shapes from data source "
			+ data_source
		)

	def __len__(self):
		return len(self.data_names)

	def __getitem__(self, idx):
		return self.data_voxels[idx], self.data_points[idx], self.data_names[idx], idx

class VoxelSamples(torch.utils.data.Dataset):
	def __init__(
		self,
		data_source
	):
		print('data source', data_source)
		self.data_source = data_source
		print('class Samples from voxels')

		name_file = os.path.join(self.data_source, 'test_names.npz')
		npz_shapes = np.load(name_file)
		self.data_names = npz_shapes['test_names']
		filename_voxels = os.path.join(self.data_source, 'voxel2mesh.hdf5')
		
		data_dict = h5py.File(filename_voxels, 'r')
		data_voxels = torch.from_numpy(data_dict['voxels'][:]).float()

		self.data_voxels = data_voxels.squeeze(-1).unsqueeze(1)
		self.data_points = torch.from_numpy(data_dict['points'][:]).float()
		self.data_points[:, :, :3] = (self.data_points[:, :, :3] + 0.5)/64-0.5
		data_dict.close()
			
		print('load voxels shape, ', self.data_voxels.shape)
		print('load points shape, ', self.data_points.shape)
		logging.debug(
			"using "
			+ str(len(self.data_voxels))
			+ " shapes from data source "
			+ data_source
		)

	def __len__(self):
		return len(self.data_voxels)

	def __getitem__(self, idx):
		return self.data_voxels[idx], self.data_points[idx], self.data_names[idx], idx


class GTSamples(torch.utils.data.Dataset):
	def __init__(
		self,
		data_source,
		grid_sample = 0,
		test_flag = False,
		shapenet_flag = False
	):
		print('data source', data_source)
		self.data_source = data_source
		print('class Samples from GT meshes')
		
		if test_flag:
			filename_shapes = os.path.join(self.data_source, 'ae_test.hdf5')
			name_file = os.path.join(self.data_source, 'test_names.npz')
			npz_shapes = np.load(name_file)
			self.data_names = npz_shapes['test_names']
		else:
			name_file = os.path.join(self.data_source, 'train_names.npz')
			npz_shapes = np.load(name_file)
			filename_shapes = os.path.join(self.data_source, 'ae_train.hdf5')
			self.data_names = npz_shapes['train_names']
		if not shapenet_flag:
			#ABC dataset
			data_dict = h5py.File(filename_shapes, 'r')
			data_voxels = torch.from_numpy(data_dict['voxels'][:]).float()
			self.data_voxels = data_voxels.squeeze(-1).unsqueeze(1)
			self.data_points = torch.from_numpy(data_dict['points_'+str(grid_sample)][:]).float()
		else:
			#ShapeNet dataset
			data_dict = h5py.File(filename_shapes, 'r')
			data_voxels = torch.from_numpy(data_dict['voxels'][:]).float()
			self.data_voxels = data_voxels.squeeze(-1).unsqueeze(1)
			data_points = torch.from_numpy(data_dict['points_'+str(grid_sample)][:]).float()
			data_points = (data_points+0.5)/256-0.5
			data_values = torch.from_numpy(data_dict['values_'+str(grid_sample)][:]).float()
			self.data_points = torch.cat([data_points, data_values], 2)
		print('load voxels shape, ', self.data_voxels.shape)
		print('load points shape, ', self.data_points.shape)
		logging.debug(
			"using "
			+ str(len(self.data_voxels))
			+ " shapes from data source "
			+ data_source
		)

	def __len__(self):
		return len(self.data_voxels)

	def __getitem__(self, idx):
		return self.data_voxels[idx], self.data_points[idx], self.data_names[idx], idx
