import torch.nn as nn
import torch
import torch.nn.functional as F

FLOAT_EPS = torch.finfo(torch.float32).eps

class Myminclamp(torch.autograd.Function):
	"""
	We can implement our own custom autograd Functions by subclassing
	torch.autograd.Function and implementing the forward and backward passes
	which operate on Tensors.
	"""

	@staticmethod
	def forward(ctx, input):
		"""
		In the forward pass we receive a Tensor containing the input and return
		a Tensor containing the output. ctx is a context object that can be used
		to stash information for backward computation. You can cache arbitrary
		objects for use in the backward pass using the ctx.save_for_backward method.
		"""
		ctx.save_for_backward(input)
		return input.clamp(min=0)

	@staticmethod
	def backward(ctx, grad_output):
		"""
		In the backward pass we receive a Tensor containing the gradient of the loss
		with respect to the output, and we need to compute the gradient of the loss
		with respect to the input.
		"""
		input, = ctx.saved_tensors
		grad_input = grad_output.clone()
		#grad_input[input < 0] = 0.001*grad_input[input < 0].clamp(max=0)
		grad_input = grad_input*(input >= 0).float() + 0.001* (grad_input * (input < 0).float()).clamp(max=0)
		return grad_input

class Mymaxclamp(torch.autograd.Function):
	"""
	We can implement our own custom autograd Functions by subclassing
	torch.autograd.Function and implementing the forward and backward passes
	which operate on Tensors.
	"""

	@staticmethod
	def forward(ctx, input):
		"""
		In the forward pass we receive a Tensor containing the input and return
		a Tensor containing the output. ctx is a context object that can be used
		to stash information for backward computation. You can cache arbitrary
		objects for use in the backward pass using the ctx.save_for_backward method.
		"""
		ctx.save_for_backward(input)
		return input.clamp(max=1)#torch.clamp(x, max=1)

	@staticmethod
	def backward(ctx, grad_output):
		"""
		In the backward pass we receive a Tensor containing the gradient of the loss
		with respect to the output, and we need to compute the gradient of the loss
		with respect to the input.
		"""
		input, = ctx.saved_tensors
		grad_input = grad_output.clone()
		grad_input = grad_input *(input <= 1).float() + 0.001*(grad_input * (input > 1).float()).clamp(min=0)
		return grad_input

class Generator(nn.Module):
	def __init__(self, p_dim=1024, c_dim=64):
		super(Generator, self).__init__()
		self.p_dim = p_dim
		self.c_dim = c_dim
		self.half_c_dim = int(c_dim/2)
		#intersection layer
		convex_layer_weights = torch.zeros((self.p_dim, self.c_dim))
		#union layer
		concave_layer_weights = torch.zeros((self.c_dim, 1))

		self.convex_layer_weights = nn.Parameter(convex_layer_weights)
		self.concave_layer_weights = nn.Parameter(concave_layer_weights)
		nn.init.normal_(self.convex_layer_weights, mean=0.0, std=0.02)
		nn.init.normal_(self.concave_layer_weights, mean=1e-5, std=0.02)

	def forward(self, points, primitives, phase, is_leaky=False):
		min_clamp = Myminclamp.apply
		max_clamp = Mymaxclamp.apply
		ones = torch.ones(points.size(0), points.size(1), 1).cuda()
		pointsx = torch.cat([points**2, points, ones], 2)
		if phase==0: #S+
			#level 1 D=relu(xP)
			#points b*x*7
			#primitives b*7*P
			h1 = torch.matmul(pointsx, primitives)
			if is_leaky:
				h1 = min_clamp(h1)
			else:
				h1 = torch.clamp(h1, min=0)
			#D b*x*P, inside 0, outside 1

			#level 2 C=1-DT
			h2 = torch.matmul(h1, self.convex_layer_weights)
			if is_leaky:
				h2 = max_clamp(min_clamp(1 - h2))
			else:
				h2 = torch.clamp(1-h2, min=0, max=1)
			#C b*x*C, inside 1, outside 0
			#level 3 a+=CW
			h3_1 = torch.matmul(h2[:, :, :self.half_c_dim], self.concave_layer_weights[:self.half_c_dim, :])
			h3_0 = torch.matmul(h2[:, :, self.half_c_dim:], self.concave_layer_weights[self.half_c_dim:, :])
			#a+ inside 1 outside 0-1
			h3 = torch.min(h3_1, (1-h3_0))

			return h3_1.squeeze(-1), h3_0.squeeze(-1), h3.squeeze(-1), h2
		elif phase==1:
			h1 = torch.matmul(pointsx, primitives)
			if is_leaky:
				h1 = min_clamp(h1)
			else:
				h1 = torch.clamp(h1, min=0)

			h2 = torch.matmul(h1, self.convex_layer_weights)
			
			#a*
			h3_1 = torch.min(h2[:, :, :self.half_c_dim], dim=2)[0]
			h3_0 = torch.min(h2[:, :, self.half_c_dim:], dim=2)[0]

			h3 = torch.max(h3_1, 0.2-h3_0)
			#reverse occ for the same threshold and surface normal orientation after MC 
			h3 = 1 - h3
			h2 = 1 - h2
			return h3_1, h3_0, h3, h2
		elif phase==2:
			h1 = torch.matmul(pointsx, primitives)
			if is_leaky:
				h1 = min_clamp(h1)
			else:
				h1 = torch.clamp(h1, min=0)
			#binary selection matrix
			h2 = torch.matmul(h1, (self.convex_layer_weights>0.01).float())
			
			h3_1 = torch.min(h2[:, :, :self.half_c_dim], dim=2)[0]
			h3_0 = torch.min(h2[:, :, self.half_c_dim:], dim=2)[0]

			h3 = torch.max(h3_1, 0.2-h3_0)
			h3 = 1 - h3
			h2 = 1 - h2
			return h3_1, h3_0, h3, h2
		else:
			print("Congrats you got an error!")
			print("generator.phase should be in [0,1,2,3], got", self.phase)
			exit(0)

class Encoder(nn.Module):
	def __init__(self, ef_dim=32):
		super(Encoder, self).__init__()
		self.ef_dim = ef_dim
		self.conv_1 = nn.Conv3d(1, self.ef_dim, 4, stride=2, padding=1, bias=True)
		self.conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim*2, 4, stride=2, padding=1, bias=True)
		self.conv_3 = nn.Conv3d(self.ef_dim*2, self.ef_dim*4, 4, stride=2, padding=1, bias=True)
		self.conv_4 = nn.Conv3d(self.ef_dim*4, self.ef_dim*8, 4, stride=2, padding=1, bias=True)
		self.conv_5 = nn.Conv3d(self.ef_dim*8, self.ef_dim*8, 4, stride=1, padding=0, bias=True)
		nn.init.xavier_uniform_(self.conv_1.weight)
		nn.init.constant_(self.conv_1.bias,0)
		nn.init.xavier_uniform_(self.conv_2.weight)
		nn.init.constant_(self.conv_2.bias,0)
		nn.init.xavier_uniform_(self.conv_3.weight)
		nn.init.constant_(self.conv_3.bias,0)
		nn.init.xavier_uniform_(self.conv_4.weight)
		nn.init.constant_(self.conv_4.bias,0)
		nn.init.xavier_uniform_(self.conv_5.weight)
		nn.init.constant_(self.conv_5.bias,0)


	def forward(self, inputs, is_training=False):
		d_1 = self.conv_1(inputs)
		d_1 = F.leaky_relu(d_1, negative_slope=0.01, inplace=True)

		d_2 = self.conv_2(d_1)
		d_2 = F.leaky_relu(d_2, negative_slope=0.01, inplace=True)
		
		d_3 = self.conv_3(d_2)
		d_3 = F.leaky_relu(d_3, negative_slope=0.01, inplace=True)
		#print('d_3', d_3.shape)
		d_4 = self.conv_4(d_3)#32 256 
		#print('d_4', d_4.shape)
		d_4 = F.leaky_relu(d_4, negative_slope=0.01, inplace=True)

		d_5 = self.conv_5(d_4)
		d_5 = d_5.view(-1, self.ef_dim*8)
		d_5 = F.leaky_relu(d_5, negative_slope=0.01, inplace=True)

		return d_5

class Decoder(nn.Module):
	def __init__(self, ef_dim=32, p_dim=1024):
		super(Decoder, self).__init__()
		self.ef_dim = ef_dim
		self.p_dim = p_dim
		self.linear_1 = nn.Linear(self.ef_dim*8, self.ef_dim*16, bias=True)
		self.linear_2 = nn.Linear(self.ef_dim*16, self.p_dim*7, bias=True)
		nn.init.xavier_uniform_(self.linear_1.weight)
		nn.init.constant_(self.linear_1.bias,0)
		nn.init.xavier_uniform_(self.linear_2.weight)
		nn.init.constant_(self.linear_2.bias,0)

	def forward(self, inputs):
		l1 = self.linear_1(inputs)
		l1 = F.leaky_relu(l1, negative_slope=0.01, inplace=True)

		l2 = self.linear_2(l1)
		l2 = l2.view(-1, 7, self.p_dim)
		l2 = torch.cat([torch.abs(l2[:, :3, :]), l2[:, 3:, :]], 1)
		return l2
