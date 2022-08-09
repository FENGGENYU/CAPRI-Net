import argparse
import json
import logging
import os
import random
import time
import torch

import utils
import utils.workspace as ws
import torch.utils.data as data_utils
import numpy as np

if __name__ == "__main__":

	arg_parser = argparse.ArgumentParser(
		description="test trained model"
	)
	arg_parser.add_argument(
		"--experiment",
		"-e",
		dest="experiment_directory",
		required=True,
		help="The experiment directory which includes specifications and saved model "
		+ "files to use for reconstruction",
	)
	arg_parser.add_argument(
		"--checkpoint",
		"-c",
		dest="checkpoint",
		default="latest",
		help="The checkpoint weights to use. This can be a number indicated an epoch "
		+ "or 'latest' for the latest weights (this is the default)",
	)
	arg_parser.add_argument(
		"--grid_sample",
		dest="grid_sample",
		default=64,
		help="dataset option",
	)
	arg_parser.add_argument(
		"--start",
		dest="start",
		default=0,
		help="start shape index",
	)
	arg_parser.add_argument(
		"--end",
		dest="end",
		default=1,
		help="end shape index",
	)
	arg_parser.add_argument(
		"--mc_threshold",
		dest="mc_threshold",
		default=0.9,
		help="marching cube threshold",
	)
	arg_parser.add_argument(
		"--csg",
		dest="csg",
		action="store_true",
		help="output csg mesh.",
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
	arg_parser.add_argument(
		"--shapenet",
		dest="shapenet",
		action="store_true",
		help="dataset option",
	)
	arg_parser.add_argument(
		"--test",
		dest="test_flag",
		action="store_true",
		help="test or train set shapes",
	)
	arg_parser.add_argument(
		"--code",
		dest="save_code",
		action="store_true",
		help="save shape code or not",
	)
	arg_parser.add_argument(
		"--primi_count",
		dest="primi_count",
		action="store_true",
		help="count primitives number or not",
	)
	arg_parser.add_argument(
		"--primi",
		dest="primi",
		action="store_true",
		help="output primitives or not",
	)
	arg_parser.add_argument(
		"--surface",
		dest="surface",
		action="store_true",
		help="dataset option",
	)
	arg_parser.add_argument(
		"--voxel",
		dest="voxel",
		action="store_true",
		help="dataset option",
	)
	#python test.py -p 2 -e abc_voxel --voxel -g 0 -c best_stage2 --test --start 0 --end 1 
	utils.add_common_args(arg_parser)

	args = arg_parser.parse_args()

	utils.configure_logging(args)
	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"]="%d"%int(args.gpu)
	phase = int(args.phase)
	start_index = int(args.start)
	end_index = int(args.end)
	csg = int(args.csg)
	grid_sample = int(args.grid_sample)
	mc_threshold = float(args.mc_threshold)
	specs_filename = os.path.join(args.experiment_directory, "specs.json")

	if not os.path.isfile(specs_filename):
		raise Exception(
			'The experiment directory does not include specifications file "specs.json"'
		)

	specs = json.load(open(specs_filename))

	arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder", "Generator"])
	decoder = arch.Decoder().cuda()
	generator = arch.Generator().cuda()
	logging.info("training with {} GPU(s)".format(torch.cuda.device_count()))

	data_source = specs["DataSource"]
	if args.surface:
		occ_dataset = utils.dataloader.SurfaceSamples(data_source, test_flag = args.test_flag)
	elif args.voxel:
		occ_dataset = utils.dataloader.VoxelSamples(data_source)
	else:
		occ_dataset = utils.dataloader.GTSamples(data_source, test_flag = args.test_flag)
	
	logging.debug(decoder)
	logging.debug(generator)
	
	if args.test_flag:
		ws.reconstructions_subdir = ws.reconstructions_subdir + '_test'
	
	if csg:
		#only in phase 2
		reconstruction_dir = os.path.join(
			args.experiment_directory, ws.reconstructions_subdir+'_csg'
		)
	else:
		reconstruction_dir = os.path.join(
			args.experiment_directory, ws.reconstructions_subdir, 'phase' + args.phase
		)

	if not os.path.isdir(reconstruction_dir):
		os.makedirs(reconstruction_dir)

	count = 0
	
	#start_indexes = [809, 1173, 1488, 2988, 4344, 4563, 5028, 5351, 5826, 6461, 8163, 8374]
	avarage_primitive_count = 0
	avarage_convex_count = 0

	shape_indexes = list(range(start_index, end_index))
	print('shape indexes all: ', shape_indexes)

	for index in shape_indexes:
		shapename = occ_dataset.data_names[index]
		print(f'index{index}, shape name {shapename}')
		saved_model_epoch, shape_code = ws.load_model_parameters_per_shape(
				args.experiment_directory, shapename, args.checkpoint,
				decoder, 
				generator,
				None,
			)
		print('load epoch: %d'%(saved_model_epoch))
		decoder.eval()
		generator.eval()

		start_time = time.time()
		primitives = decoder(shape_code.cuda())

		mesh_filename = os.path.join(reconstruction_dir, shapename)
		if csg:
			convex_num, primi_count = utils.cad_meshing.create_cad_mesh(
				generator, primitives, connection_t=generator.convex_layer_weights, filename = mesh_filename,
				inter_results_flag = True, final_result_flag = True
			)
			avarage_primitive_count += primi_count
			avarage_convex_count += convex_num
			count += 1
		else:
			utils.cad_meshing.create_mesh_mc(
				generator, phase, primitives.cuda(), mesh_filename, N=128, threshold=mc_threshold
			)
		logging.debug("reconstruct time: {}".format(time.time() - start_time))

	if args.primi_count:
		print('avarage_primitive_count, {}, avarage_convex_count  {}, '.format(avarage_primitive_count/count, avarage_convex_count/count))

