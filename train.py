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

def main_function(experiment_directory, continue_from, phase, grid_sample, leaky, sample_surface, shapenet_flag):

	init_seeds()

	logging.debug("running " + experiment_directory)

	specs = ws.load_experiment_specifications(experiment_directory)
	print(specs["Description"])
	logging.info("Experiment description: \n" + specs["Description"])

	data_source = specs["DataSource"]

	arch = __import__("networks." + specs["NetworkArch"], fromlist=["Encoder", "Decoder", "Generator"])

	checkpoints = list(
		range(
			specs["SnapshotFrequency"],
			specs["NumEpochs"] + 1,
			specs["SnapshotFrequency"],
		)
	)
	
	for checkpoint in specs["AdditionalSnapshots"]:
		checkpoints.append(checkpoint)
	checkpoints.sort()
	print(checkpoints)
	lr_schedules = get_learning_rate_schedules(specs)

	def save_latest(epoch):

		ws.save_model_parameters(experiment_directory, "latest.pth", encoder, decoder, generator, optimizer_all, epoch)
		
	def save_checkpoints(epoch):

		ws.save_model_parameters(experiment_directory, str(epoch) + ".pth", encoder, decoder, generator, optimizer_all, epoch)
	
	def save_checkpoints_best(epoch):

		ws.save_model_parameters(experiment_directory, "best_stage%d_%d.pth"%(phase, grid_sample), encoder, decoder, generator, optimizer_all, epoch)
		
	def signal_handler(sig, frame):
		logging.info("Stopping early...")
		sys.exit(0)

	def adjust_learning_rate(lr_schedules, optimizer, epoch):

		for i, param_group in enumerate(optimizer.param_groups):
			param_group["lr"] = lr_schedules[0].get_learning_rate(epoch)

	signal.signal(signal.SIGINT, signal_handler)

	#set batch size based on GPU memory size
	if grid_sample == 32:
		scene_per_batch = 24
	elif grid_sample==64:
		scene_per_batch = 24
	else:
		scene_per_batch = 24

	encoder = arch.Encoder().cuda()
	decoder = arch.Decoder().cuda()
	generator = arch.Generator().cuda()
	logging.info("training with {} GPU(s)".format(torch.cuda.device_count()))
	
	num_epochs = specs["NumEpochs"]

	if sample_surface:
		occ_dataset = utils.dataloader.SurfaceSamples(
			data_source, test_flag=False
		)
	else:
		occ_dataset = utils.dataloader.GTSamples(
			data_source, grid_sample=grid_sample, test_flag=False, shapenet_flag=shapenet_flag
		)

	data_loader = data_utils.DataLoader(
		occ_dataset,
		batch_size=scene_per_batch,
		shuffle=True,
		num_workers=4
	)

	num_scenes = len(occ_dataset)

	logging.info("There are {} shapes".format(num_scenes))

	logging.debug(decoder)
	logging.debug(encoder)
	logging.debug(generator)

	optimizer_all = torch.optim.Adam(
		[
			{
				"params": decoder.parameters(),
				"lr": lr_schedules[0].get_learning_rate(0),
				"betas": (0.5, 0.999),
			},
			{
				"params": encoder.parameters(),
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
	if continue_from is not None:

		logging.info('continuing from "{}"'.format(continue_from))

		model_epoch = ws.load_model_parameters(
			experiment_directory, continue_from, 
			encoder,
			decoder,
			generator,
			optimizer_all
		)

		start_epoch = model_epoch + 1
		logging.debug("loaded")

	logging.info("starting from epoch {}".format(start_epoch))
	logging.info(f"Training, expriment {experiment_directory}, batch size {scene_per_batch}, phase {phase}, \
		grid_sample {grid_sample}, leaky {leaky} , shapenet_flag {shapenet_flag}, no weight {no_weight}")
	decoder.train()
	encoder.train()
	generator.train()

	start_time = time.time()
	start_epoch = 0
	last_epoch_time = 0

	point_batch_size = 16*16*16*2

	load_point_batch_size = occ_dataset.data_points.shape[1]
	point_batch_num = int(load_point_batch_size/point_batch_size)
	print('point batch num, ', point_batch_num)

	best_loss = 999

	for epoch in range(start_epoch, start_epoch + num_epochs):
		
		adjust_learning_rate(lr_schedules, optimizer_all, epoch - start_epoch)

		avarage_left_loss = 0
		avarage_right_loss = 0
		avarage_total_loss = 0
		avarage_num = 0
		iters = 0
		for voxels, occ_data, shape_names, indices in data_loader:

			voxels = voxels.cuda()
			occ_data = occ_data.cuda()
			iters += 1

			which_batch = torch.randint(point_batch_num+1, (1,))
			if which_batch == point_batch_num:
				xyz = occ_data[:,-point_batch_size:, :3]
				occ_gt = occ_data[:,-point_batch_size:, 3]
			else:
				xyz = occ_data[:,which_batch*point_batch_size:(which_batch+1)*point_batch_size, :3]
				occ_gt = occ_data[:,which_batch*point_batch_size:(which_batch+1)*point_batch_size, 3]
			
			optimizer_all.zero_grad()

			shape_code = encoder(voxels)
			primitives = decoder(shape_code)

			G_left, G_right, net_out, net_out_convexes = generator(xyz, primitives, phase, leaky)
			value_loss_left, value_loss_right, total_loss = loss(shapenet_flag, phase, G_left, G_right, occ_gt, 
				generator.concave_layer_weights, generator.convex_layer_weights, no_weight)
					
			total_loss.backward()
			optimizer_all.step()

			avarage_left_loss += value_loss_left.detach().item()
			avarage_right_loss += value_loss_right.detach().item()
			avarage_total_loss += total_loss.detach().item()
			avarage_num += 1

		if (epoch+1) % 1 == 0:
			seconds_elapsed = time.time() - start_time
			ava_epoch_time = (seconds_elapsed - last_epoch_time)/10
			left_time = ava_epoch_time*(num_epochs+ start_epoch- epoch)/3600
			last_epoch_time = seconds_elapsed
			left_loss = avarage_left_loss/avarage_num
			right_loss = avarage_right_loss/avarage_num
			t_loss = avarage_total_loss/avarage_num
			logging.debug("epoch = {}/{} err_left = {:.6f}, err_right = {:.6f}, \
				total_loss={:.6f}, 1 epoch time ={:.6f}, left time={:.6f}".format(epoch, 
				num_epochs+start_epoch, left_loss, right_loss, t_loss, ava_epoch_time, left_time))

			if t_loss < best_loss:
				print('best loss updated, ', t_loss)
				save_checkpoints_best(epoch)
				best_loss = t_loss

		if (epoch-start_epoch+1) in checkpoints:
			save_checkpoints(epoch)


if __name__ == "__main__":

	import argparse

	arg_parser = argparse.ArgumentParser(description="Train a Network")
	arg_parser.add_argument(
		"--experiment",
		"-e",
		dest="experiment_directory",
		required=True,
		help="The experiment directory. This directory should include "
		+ "experiment specifications in 'specs.json', and logging will be "
		+ "done in this directory as well.",
	)
	arg_parser.add_argument(
		"--continue",
		"-c",
		dest="continue_from",
		help="A snapshot to continue from. This can be 'latest' to continue"
		+ "from the latest running snapshot, or an integer corresponding to "
		+ "an epochal snapshot.",
	)
	#soft min max or not
	arg_parser.add_argument(
		"--leaky",
		dest="leaky",
		action="store_false",
		help="soft min max",
	)
	arg_parser.add_argument(
		"--grid_sample",
		dest="grid_sample",
		default=64,
		help="sample points resolution option",
	)
	arg_parser.add_argument(
		"--surface",
		dest="surface",
		action="store_true",
		help="point cloud option",
	)
	arg_parser.add_argument(
		"--shapenet_flag",
		dest="shapenet_flag",
		action="store_true",
		help="voxel option",
	)
	arg_parser.add_argument(
		"--gpu",
		"-g",
		dest="gpu",
		required=True,
		help="gpu id",
	)
	arg_parser.add_argument(
		"--phase",
		"-p",
		dest="phase",
		required=True,
		help="phase stage",
	)
	utils.add_common_args(arg_parser)

	args = arg_parser.parse_args()

	utils.configure_logging(args)
	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"]="%d"%int(args.gpu)
	print('gpu: ,', int(args.gpu))
	main_function(args.experiment_directory, args.continue_from, int(args.phase), \
		int(args.grid_sample), args.leaky, args.surface, args.shapenet_flag)
