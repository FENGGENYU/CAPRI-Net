import logging
import torch
import numpy as np
import plyfile
import os

def add_common_args(arg_parser):
	arg_parser.add_argument(
		"--debug",
		dest="debug",
		default=True,
		action="store_true",
		help="If set, debugging messages will be printed",
	)
	arg_parser.add_argument(
		"--quiet",
		"-q",
		dest="quiet",
		default=True,
		action="store_true",
		help="If set, only warnings will be printed",
	)
	arg_parser.add_argument(
		"--log",
		dest="logfile",
		default=None,
		help="If set, the log will be saved using the specified filename.",
	)


def configure_logging(args):
	logger = logging.getLogger()
	if args.debug:
		logger.setLevel(logging.DEBUG)
	elif args.quiet:
		logger.setLevel(logging.WARNING)
	else:
		logger.setLevel(logging.INFO)
	logger_handler = logging.StreamHandler()
	formatter = logging.Formatter("CAPRI - %(levelname)s - %(message)s")
	logger_handler.setFormatter(formatter)
	logger.addHandler(logger_handler)

	if args.logfile is not None:
		file_logger_handler = logging.FileHandler(args.logfile)
		file_logger_handler.setFormatter(formatter)
		logger.addHandler(file_logger_handler)

def save_obj_data(filename, vertex, face):
	"""
	saves only vertices and faces
	"""

	numver = vertex.shape[0]
	numfac = face.shape[0]
	with open(filename, 'w') as f:
		f.write('# %d vertices, %d faces'%(numver, numfac))
		f.write('\n')
		for v in vertex:
			f.write('v %f %f %f' %(v[0], v[1], v[2]))
			f.write('\n')
		for F in face:
			f.write('f %d %d %d' %(F[0]+1, F[1]+1, F[2]+1))
			f.write('\n')
			
def write_ply_triangle(ply_filename_out, verts, faces):
	
	num_verts = verts.shape[0]
	num_faces = faces.shape[0]

	verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

	for i in range(0, num_verts):
		verts_tuple[i] = tuple(verts[i, :])

	faces_building = []
	for i in range(0, num_faces):
		faces_building.append(((faces[i, :].tolist(),)))
	faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

	el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
	el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

	ply_data = plyfile.PlyData([el_verts, el_faces])
	logging.debug("saving mesh to %s" % (ply_filename_out))
	ply_data.write(ply_filename_out)

def write_ply_point_color(name, vertices, colors):
	fout = open(name, 'w')
	fout.write("ply\n")
	fout.write("format ascii 1.0\n")
	fout.write("element vertex "+str(len(vertices))+"\n")
	fout.write("property float x\n")
	fout.write("property float y\n")
	fout.write("property float z\n")
	fout.write("property uchar red\n")
	fout.write("property uchar green\n")
	fout.write("property uchar blue\n")
	fout.write("end_header\n")
	for ii in range(len(vertices)):
		fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+" "+str(colors[ii,0])+" "+str(colors[ii,1])+" "+str(colors[ii,2])+"\n")
	fout.close()

def write_ply_point_normal(name, vertices, normals=None):
	fout = open(name, 'w')
	fout.write("ply\n")
	fout.write("format ascii 1.0\n")
	fout.write("element vertex "+str(len(vertices))+"\n")
	fout.write("property float x\n")
	fout.write("property float y\n")
	fout.write("property float z\n")
	fout.write("property float nx\n")
	fout.write("property float ny\n")
	fout.write("property float nz\n")
	fout.write("end_header\n")
	if normals is None:
		for ii in range(len(vertices)):
			fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+" "+str(vertices[ii,3])+" "+str(vertices[ii,4])+" "+str(vertices[ii,5])+"\n")
	else:
		for ii in range(len(vertices)):
			fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+" "+str(normals[ii,0])+" "+str(normals[ii,1])+" "+str(normals[ii,2])+"\n")
	fout.close()

def write_ply_polygon(name, vertices, polygons):
	fout = open(name, 'w')
	fout.write("ply\n")
	fout.write("format ascii 1.0\n")
	fout.write("element vertex "+str(len(vertices))+"\n")
	fout.write("property float x\n")
	fout.write("property float y\n")
	fout.write("property float z\n")
	fout.write("element face "+str(len(polygons))+"\n")
	fout.write("property list uchar int vertex_index\n")
	fout.write("end_header\n")
	for ii in range(len(vertices)):
		fout.write(str(vertices[ii][0])+" "+str(vertices[ii][1])+" "+str(vertices[ii][2])+"\n")
	for ii in range(len(polygons)):
		fout.write(str(len(polygons[ii])))
		for jj in range(len(polygons[ii])):
			fout.write(" "+str(polygons[ii][jj]))
		fout.write("\n")
	fout.close()
	
def save_xyz_data(filename, ver):
	"""writes the points to a xyz file"""
   
	head, name = os.path.split(filename)
	write_path = os.path.join(head, name.split('.')[0] + '.xyz')
	np.savetxt(write_path, ver)