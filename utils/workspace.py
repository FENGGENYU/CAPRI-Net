import json
import os
import torch

model_params_subdir = "ModelParameters"
latent_codes_subdir = "LatentCodes"
logs_filename = "Logs.pth"
reconstructions_subdir = "Reconstructions"
specifications_filename = "specs.json"

def load_experiment_specifications(experiment_directory):

	filename = os.path.join(experiment_directory, specifications_filename)

	if not os.path.isfile(filename):
		raise Exception(
			"The experiment directory ({}) does not include specifications file "
			+ '"specs.json"'.format(experiment_directory)
		)

	return json.load(open(filename))

def save_model_parameters(experiment_directory, filename, encoder, decoder, generator, opt, epoch):

	model_params_dir = get_model_params_dir(experiment_directory, True)

	torch.save(
		{"epoch": epoch,
		"encoder_state_dict": encoder.state_dict(),
		"decoder_state_dict": decoder.state_dict(),
		"generator_state_dict": generator.state_dict(),
		"opt_state_dict": opt.state_dict()},
		os.path.join(model_params_dir, filename),
	)
	
def load_model_parameters(experiment_directory, checkpoint, encoder, decoder, generator, opt):

	filename = os.path.join(
		experiment_directory, model_params_subdir, checkpoint + ".pth"
	)

	if not os.path.isfile(filename):
		raise Exception('model state dict "{}" does not exist'.format(filename))

	data = torch.load(filename)

	encoder.load_state_dict(data["encoder_state_dict"])
	decoder.load_state_dict(data["decoder_state_dict"])
	generator.load_state_dict(data["generator_state_dict"])
	if opt is not None:
		opt.load_state_dict(data["opt_state_dict"])
	return data["epoch"]

def save_model_parameters_per_shape(experiment_directory, shapename, filename, decoder, generator, shape_code, opt, epoch):

	model_params_dir = get_model_params_dir(experiment_directory, True)
	model_params_dir = get_model_params_dir_shapename(model_params_dir, shapename, True)

	torch.save(
		{"epoch": epoch,
		"plane_state_dict": shape_code,
		"decoder_state_dict": decoder.state_dict(),
		"generator_state_dict": generator.state_dict(),
		"opt_state_dict": opt.state_dict()}, 
		os.path.join(model_params_dir, filename)
	)

def load_model_parameters_per_shape(experiment_directory, shapename, checkpoint, decoder, generator, opt):

	filename = os.path.join(
		experiment_directory, model_params_subdir, shapename, checkpoint + ".pth"
	)

	if not os.path.isfile(filename):
		raise Exception('model state dict "{}" does not exist'.format(filename))

	data = torch.load(filename)

	#test stage no opt
	if opt is not None:
		opt.load_state_dict(data["opt_state_dict"])
	decoder.load_state_dict(data["decoder_state_dict"])
	generator.load_state_dict(data["generator_state_dict"])
	return data["epoch"], data["plane_state_dict"]


def get_model_params_dir(experiment_dir, create_if_nonexistent=False):

	dir = os.path.join(experiment_dir, model_params_subdir)

	if create_if_nonexistent and not os.path.isdir(dir):
		os.makedirs(dir)

	return dir

def get_model_params_dir_shapename(experiment_dir, shapename, create_if_nonexistent=False):

	dir = os.path.join(experiment_dir, shapename)

	if create_if_nonexistent and not os.path.isdir(dir):
		os.makedirs(dir)

	return dir
