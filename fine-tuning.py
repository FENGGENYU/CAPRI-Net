import torch
import torch.utils.data as data_utils
import signal
import sys
import os
import logging
import math
import json
import time

import utils
import utils.workspace as ws
from networks.losses import loss
import numpy as np

class LearningRateSchedule:
	def get_learning_rate(self, epoch):
		pass


class ConstantLearningRateSchedule(LearningRateSchedule):
	def __init__(self, value):
		self.value = value

	def get_learning_rate(self, epoch):
		return self.value


class StepLearningRateSchedule(LearningRateSchedule):
	def __init__(self, initial, interval, factor):
		self.initial = initial
		self.interval = interval
		self.factor = factor

	def get_learning_rate(self, epoch):

		return self.initial * (self.factor ** (epoch // self.interval))


class WarmupLearningRateSchedule(LearningRateSchedule):
	def __init__(self, initial, warmed_up, length):
		self.initial = initial
		self.warmed_up = warmed_up
		self.length = length

	def get_learning_rate(self, epoch):
		if epoch > self.length:
			return self.warmed_up
		return self.initial + (self.warmed_up - self.initial) * epoch / self.length


def get_learning_rate_schedules(specs):

	schedule_specs = specs["LearningRateSchedule"]

	schedules = []

	for schedule_specs in schedule_specs:

		if schedule_specs["Type"] == "Step":
			schedules.append(
				StepLearningRateSchedule(
					schedule_specs["Initial"],
					schedule_specs["Interval"],
					schedule_specs["Factor"],
				)
			)
		elif schedule_specs["Type"] == "Warmup":
			schedules.append(
				WarmupLearningRateSchedule(
					schedule_specs["Initial"],
					schedule_specs["Final"],
					schedule_specs["Length"],
				)
			)
		elif schedule_specs["Type"] == "Constant":
			schedules.append(ConstantLearningRateSchedule(schedule_specs["Value"]))

		else:
			raise Exception(
				'no known learning rate schedule of type "{}"'.format(
					schedule_specs["Type"]
				)
			)

	return schedules

def get_spec_with_default(specs, key, default):
	try:
		return specs[key]
	except KeyError:
		return default

def init_seeds(seed=0):
	torch.manual_seed(seed) # sets the seed for generating random numbers.
	torch.cuda.manual_seed(seed) # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
	torch.cuda.manual_seed_all(seed) # Sets the seed for generating random numbers on all GPUs. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
	#torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True

def main_function(start_index, end_index, experiment_directory, 
	grid_sample, leaky, test_flag, sample_surface, sample_voxel, 
	shapenet_flag, no_weight):

	init_seeds()

	logging.debug("running " + experiment_directory)

	specs = ws.load_experiment_specifications(experiment_directory)
	logging.info("Experiment description: \n" + specs["Description"])

	data_source = specs["DataSource"]

	arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder", "Generator"])

	lr_schedules = get_learning_rate_schedules(specs)

	def save_checkpoints_best(epoch):

		ws.save_model_parameters_per_shape(experiment_directory, shapename, "best_stage%d.pth"%(phase), decoder, generator, shape_code, optimizer_all, epoch)
	
	def save_checkpoints(epoch):

		ws.save_model_parameters_per_shape(experiment_directory, shapename, "last_%d.pth"%(phase), decoder, generator, shape_code, optimizer_all, epoch)
	
	def signal_handler(sig, frame):
		logging.info("Stopping early...")
		sys.exit(0)

	def adjust_learning_rate(lr_schedules, optimizer, epoch):

		for i, param_group in enumerate(optimizer.param_groups):
			param_group["lr"] = lr_schedules[0].get_learning_rate(epoch)

	signal.signal(signal.SIGINT, signal_handler)

	encoder = arch.Encoder().cuda()
	decoder = arch.Decoder().cuda()
	generator = arch.Generator().cuda()

	log_frequency = get_spec_with_default(specs, "LogFrequency", 50)

	if sample_surface:
		occ_dataset = utils.dataloader.SurfaceSamples(data_source)
	elif sample_voxel:
		occ_dataset = utils.dataloader.VoxelSamples(data_source)
	else:
		occ_dataset = utils.dataloader.GTSamples(data_source)

	logging.debug(decoder)
	logging.debug(generator)

	#shapenet category start index
	#plane, bench, cabinet, car
	#0, 809, 1173, 1488
	#chair, display, lamp, speaker
	#2988, 4344, 4563, 5027
	#riffer couch, table, phone, vessel
	#5351, 5826, 6461, 8163, 8374

	if shapenet_flag:
		cate_indexes = [0, 809, 1173, 1488, 2988, 4344, 4563, 5027, 5351, 5826, 6461, 8163, 8374]
		cate_shape_indexes = []
		for i in cate_indexes:
			cate_shape_indexes = cate_shape_indexes + list(range(i, i+100))
		shape_indexes = cate_shape_indexes[start_index:end_index]
	else:
		shape_indexes = list(range(start_index, end_index))
	print('shape indexes in hdf5: ', shape_indexes)

	logging.info("There are {} shapes to be fine-tuned".format(len(shape_indexes)))
	#updated here

	load_point_batch_size = occ_dataset.data_points.shape[1]
	point_batch_num = 4
	point_batch_size = int(load_point_batch_size/point_batch_num)
	print('point batch num, ', point_batch_num)
	print('point_batch_size, ', point_batch_size)
	iterations = int(10*point_batch_num)
	print('iterations per epoch, ', iterations)

	epoches_each_stage = 200 #more is better, optional 500 is more stable than 300
	stages = [0, 1, 2]

	logging.info(f"test {test_flag}, expriment {experiment_directory}, grid_sample {grid_sample}, leaky {leaky}, shapenet {shapenet_flag} noweight {no_weight}")

	for index in shape_indexes:

		optimizer_all = torch.optim.Adam(
			[
				{
					"params": decoder.parameters(),
					"lr": lr_schedules[0].get_learning_rate(0),
					"betas": (0.5, 0.999),
				},
				{
					"params": generator.parameters(),
					"lr": lr_schedules[0].get_learning_rate(0),
					"betas": (0.5, 0.999),
				},
			]
		)
		print('fine-tuning shape index', index)
		
		shapename = occ_dataset.data_names[index]
		print('fine-tuning shape name', shapename)
		occ_data = occ_dataset.data_points[index].cuda()
		occ_data = occ_data.unsqueeze(0)
		voxels = occ_dataset.data_voxels[index].unsqueeze(0).cuda()

		for phase in stages:
			print('phase , ', phase)
			num_epochs = epoches_each_stage
			if phase == 0:
				continue_from = 'initial'
				logging.info('continuing from "{}"'.format(continue_from))
				model_epoch = ws.load_model_parameters(
					experiment_directory, continue_from, 
					encoder,
					decoder,
					generator,
					None
				)
				shape_code = encoder(voxels)
				shape_code = shape_code.detach().cpu().numpy()

				shape_code = torch.from_numpy(shape_code)
				print('shape_code loaded, ', shape_code.shape)
				start_epoch = model_epoch +1 
				logging.debug("weights loaded")
			else:
				continue_from = "last_%d"%(phase-1)
				logging.info('continuing from "{}"'.format(continue_from))
				model_epoch, shape_code = ws.load_model_parameters_per_shape(
					experiment_directory, shapename, continue_from, 
					decoder,
					generator,
					optimizer_all
				)
				print('shape_code loaded, ', shape_code.shape)
				start_epoch = model_epoch +1 
				logging.debug("loaded")

			shape_code.requires_grad = True

			optimizer_code = torch.optim.Adam(
				[
					{
						"params": shape_code,
						"lr": lr_schedules[0].get_learning_rate(0),
						"betas": (0.5, 0.999),
					},
				]
			)

			best_loss = 999
			start_epoch = model_epoch +1 
			logging.info("starting from epoch {}".format(start_epoch))
			
			decoder.train()
			generator.train()
			start_time = time.time()
			last_epoch_time = 0
			for epoch in range(start_epoch, start_epoch + num_epochs):
				
				adjust_learning_rate(lr_schedules, optimizer_all, epoch - start_epoch)
				adjust_learning_rate(lr_schedules, optimizer_code, epoch - start_epoch)

				avarage_left_loss = 0
				avarage_right_loss = 0
				avarage_total_loss = 0
				avarage_num = 0

				for itera in range(iterations):

					which_batch = torch.randint(point_batch_num, (1,))

					xyz = occ_data[:,which_batch*point_batch_size:(which_batch+1)*point_batch_size, :3]
					occ_gt = occ_data[:,which_batch*point_batch_size:(which_batch+1)*point_batch_size, 3]
					
					optimizer_all.zero_grad()
					optimizer_code.zero_grad()

					primitives = decoder(shape_code.cuda())
					
					G_left, G_right, net_out, net_out_convexes = generator(xyz, primitives, phase, leaky)
					value_loss_left, value_loss_right, total_loss = loss(phase, G_left, G_right, occ_gt, 
						generator.concave_layer_weights, generator.convex_layer_weights, no_weight)
					
					total_loss.backward()
					optimizer_all.step()
					optimizer_code.step()

					avarage_left_loss += value_loss_left.detach().item()
					avarage_right_loss += value_loss_right.detach().item()
					avarage_total_loss += total_loss.detach().item()
					avarage_num += 1

				if (epoch- start_epoch +1) % 10 == 0:
					end = time.time()
					seconds_elapsed = end - start_time
					ava_epoch_time = (seconds_elapsed - last_epoch_time)/10
					left_time = ava_epoch_time*(num_epochs+ start_epoch- epoch)/60
					last_epoch_time = seconds_elapsed
					left_loss = avarage_left_loss/avarage_num
					right_loss = avarage_right_loss/avarage_num
					t_loss = avarage_total_loss/avarage_num
					logging.debug("epoch = {}/{} err_left = {:.6f}, err_right = {:.6f}, \
						total_loss={:.6f}, 1 epoch time ={:.6f}, left time={:.6f}".format(epoch, 
						num_epochs+start_epoch, left_loss, right_loss, t_loss, ava_epoch_time, left_time))

					if t_loss < best_loss and phase == 2:
						print('best loss updated, ', t_loss)
						save_checkpoints_best(epoch)
						best_loss = t_loss
				if (epoch - start_epoch +1) % num_epochs == 0:
					save_checkpoints(epoch)

			print('stage time:, ', time.time() - start_time)

if __name__ == "__main__":
	#python fine-tuning.py -e abc_voxel --test --voxel -g 0 --start 0 --end 1
	import argparse

	arg_parser = argparse.ArgumentParser(description="Finetuning network")
	arg_parser.add_argument(
		"--experiment",
		"-e",
		dest="experiment_directory",
		required=True,
		help="The experiment directory. This directory should include "
		+ "experiment specifications in 'specs.json', and logging will be "
		+ "done in this directory as well.",
	)
	#soft min max or not
	arg_parser.add_argument(
		"--leaky",
		dest="leaky",
		action="store_false",
		help="soft min max",
	)
	arg_parser.add_argument(
		"--noweight",
		dest="noweight",
		action="store_true",
		help="make left op and right op the same weight or not option",
	)

	arg_parser.add_argument(
		"--test",
		dest="test",
		action="store_true",
		help="test or train option",
	)
	arg_parser.add_argument(
		"--grid_sample",
		dest="grid_sample",
		default=64,
		help="dataset option",
	)
	arg_parser.add_argument(
		"--start",
		dest="start_index",
		default=0,
		help="finetuning start index",
	)
	arg_parser.add_argument(
		"--end",
		dest="end_index",
		default=1,
		help="finetuning end_index",
	)
	arg_parser.add_argument(
		"--surface",
		dest="surface",
		action="store_true",
		help="point cloud option",
	)
	arg_parser.add_argument(
		"--voxel",
		dest="voxel",
		action="store_true",
		help="voxel option",
	)
	arg_parser.add_argument(
		"--shapenet_flag",
		dest="shapenet_flag",
		action="store_true",
		help="dataset option",
	)
	arg_parser.add_argument(
		"--gpu",
		"-g",
		dest="gpu",
		required=True,
		help="gpu id",
	)
	utils.add_common_args(arg_parser)

	args = arg_parser.parse_args()

	utils.configure_logging(args)
	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"]="%d"%int(args.gpu)
	print('gpu: ,', int(args.gpu))
	main_function(int(args.start_index), int(args.end_index), args.experiment_directory, 
		int(args.grid_sample), args.leaky, args.test, args.surface, args.voxel, 
		args.shapenet_flag, args.noweight)
