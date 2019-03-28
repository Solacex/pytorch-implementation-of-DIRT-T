import pretrainedmodels
import resnet
import importlib

def get_pretrained_model(arch, maintain_fc=False):
	if hasattr(resnet, arch):
		backbone = getattr(resnet, arch)()

