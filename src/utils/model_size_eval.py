from prettytable import PrettyTable
import torch
import os

def print_model_size(mdl):
    print("Model size:", end=" ")
    torch.save(mdl.state_dict(), "tmp.pt")
    print("%.2f MB" %(os.path.getsize("tmp.pt")/2**20))
    # print(os.path.getsize("tmp.pt")/1e6)
    os.remove('tmp.pt')

def get_model_size(mdl):
    torch.save(mdl.state_dict(), "tmp.pt")
    model_size = os.path.getsize("tmp.pt")/2**20
    os.remove('tmp.pt')
    return model_size


def count_parameters_detailed(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in dict(model.state_dict()).items():
        if not ('weight' in name or 'bias' in name):
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    print("More detailed parameters distribution:")
    print(table)
    return total_params

def count_parameters_short(model):
    print(sum(p.numel() for p in model.parameters()))

def get_parameters_num(model):
    return sum(p.numel() for p in model.parameters())
