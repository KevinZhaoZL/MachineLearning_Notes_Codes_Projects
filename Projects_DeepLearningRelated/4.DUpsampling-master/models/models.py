import torch

def create_model(opt, dataset=None):
    if opt.model == "DUNet":
        from models.dunet import DUNet_Solver
        model = DUNet_Solver(opt)

    if opt.model == "DUNet_sybn":
        from models.dunet_sybn import DUNet_Solver
        model = DUNet_Solver(opt)
    return model
