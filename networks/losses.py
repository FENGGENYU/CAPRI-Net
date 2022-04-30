import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

def loss(phase, G_left, G_right, trues, shape_layer, convex_layer, no_weight=False):
	#convex layer -> intersection layer
	#shape layer -> union layer
	#differ layer no weights
	if no_weight:
		a = 1
		b = 1
		c = 1
	else:
		a = 10
		b = 2.5
		c = 1
	#trues - ground truth inside-outside value for each point, inside 1, outside 0
	#loss
	if phase==0:
		#phase 0 all continuous for better convergence
		#hard differ layer
		#inside 1, outside 0-1
		inversed = 1-trues
		mask_left = torch.max(trues, (G_right<0.5).float())
		mask_right = torch.max(trues, (G_left>0.5).float())
		loss_left = torch.mean(((trues*(1-G_left.clamp(max=1)) + inversed*(G_left.clamp(min=0)))**2) * mask_left)
		loss_right = torch.mean(((inversed*(1-G_right.clamp(max=1)) + trues*(G_right.clamp(min=0)))**2) *mask_right)

		total_loss = loss_left + loss_right + torch.sum(torch.abs(shape_layer-1))
		total_loss +=  (torch.sum(torch.clamp(convex_layer-1, min=0) - torch.clamp(convex_layer, max=0)))
		
	elif phase==1:
		#phase 1 hard union layer
		inversed = 1-trues
		mask_left = torch.max(trues, (G_right>0.01).float())
		mask_right = torch.max(trues, (G_left<0.01).float())
		
		loss_left = torch.mean((inversed*(1-G_left.clamp(max=1)) + a*trues*(G_left.clamp(min=0)))*mask_left)
		loss_right = torch.mean((trues*(1-G_right.clamp(max=1))*c + b*inversed*(G_right.clamp(min=0)))*mask_right)
		total_loss = loss_left + loss_right
		total_loss +=  (torch.sum(torch.clamp(convex_layer-1, min=0) - torch.clamp(convex_layer, max=0)))

	elif phase==2:
		#phase 1 hard intersection layer
		inversed = 1-trues
		mask_left = torch.max(trues, (G_right>0.01).float())
		mask_right = torch.max(trues, (G_left<0.01).float())
		
		loss_left = torch.mean((inversed*(1-G_left.clamp(max=1)) + a*trues*(G_left.clamp(min=0)))*mask_left)
		loss_right = torch.mean((trues*(1-G_right.clamp(max=1))*c + b*inversed*(G_right.clamp(min=0)))*mask_right)
		
		total_loss = loss_left + loss_right
	else:
		print('error....................')

	return loss_left, loss_right, total_loss