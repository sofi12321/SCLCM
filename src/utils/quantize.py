import copy
import time
import torch
from torch.quantization import quantize_fx, get_default_qconfig
from utils.model_size_eval import print_model_size, count_parameters_detailed, count_parameters_short, get_model_size, get_parameters_num
from utils.test_run_classification import test_classification

def quantize_model(ft_model, test_loader_ft, device = 'cpu', backend = "fbgemm"):
    model_static_quantized = copy.deepcopy(ft_model)
    model_static_quantized.to(device)
    model_static_quantized.eval()

    qconfig_dict = {"": get_default_qconfig(backend)}
    # qconfig_dict = {"": torch.quantization.QConfig(
    #             activation=HistogramObserver,
    #             weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
    #         )}
    example_inputs = next(iter(test_loader_ft))['data']
    # Prepare
    model_prepared = quantize_fx.prepare_fx(model_static_quantized, qconfig_dict, example_inputs)
    # Calibrate - Use representative (validation) data.
    with torch.inference_mode():
        for _ in range(5):
            x = next(iter(test_loader_ft))['data']
            model_prepared(x)
    # Quantize
    model_quantized = quantize_fx.convert_fx(model_prepared)
    return model_quantized


def eval_quantization(model_quantized, test_loader_ft, num_classes, device='cpu'):
    # Evaluate size
    model_size = print_model_size(model_quantized)
    total_params = count_parameters_detailed(model_quantized)
    print()

    # Evaluate performance
    criterion = torch.nn.CrossEntropyLoss()
    if num_classes == 2:
        criterion = torch.nn.BCEWithLogitsLoss()

    start_time = time.time()
    cur_loss, cur_acc = test_classification(model_quantized, device, test_loader_ft, criterion)
    test_time = time.time() - start_time

    print("Loss:", round(cur_loss, 2))
    print("Accuracy:", round(cur_acc, 2))
    print("Processing time:", test_time, "sec")

def compare_quantization(model_init, model_quantized, test_loader_ft, num_classes, device='cpu'):
    # Evaluate size
    model_size_init = get_model_size(model_init)
    model_size = get_model_size(model_quantized)
    print("Model size changed from %.2f MB to %.2f MB" %(model_size_init, model_size) )
    print(f"The size of the model has decreased by {round(model_size_init / model_size, 2)} times")

    total_params_init = get_parameters_num(model_init)
    total_params = get_parameters_num(model_quantized)
    if total_params_init == total_params:
        print("The number of trainable parameters has not changed")
    else:
        print(f"Number of trainable parameters changed from {total_params_init} to {total_params}, or by {round(total_params_init / total_params, 2)} times")

    # Evaluate performance
    criterion = torch.nn.CrossEntropyLoss()
    if num_classes == 2:
        criterion = torch.nn.BCEWithLogitsLoss()

    start_time = time.time()
    loss_init, acc_init = test_classification(model_init, device, test_loader_ft, criterion)
    test_time_init = time.time() - start_time

    start_time = time.time()
    cur_loss, cur_acc = test_classification(model_quantized, device, test_loader_ft, criterion)
    test_time = time.time() - start_time

    print(f"Loss changed on {round(loss_init - cur_loss, 2)} times")
    print(f"Accuracy changed on {round(acc_init - cur_acc, 2)*100}%")
    
    print(f"Processing time changed by {round(test_time_init / test_time, 2)} times") 
