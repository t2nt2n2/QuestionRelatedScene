import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)

def triplet_loss(output):

	margin = 0.5
    
	losses = F.relu(output[0][:,1].reshape(-1,1) - output[1][:,1].reshape(-1,1) + margin)
	#print(output[0][:,1],output[1][:,1])
	return losses.sum()
