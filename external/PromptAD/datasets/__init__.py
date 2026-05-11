import numpy as np
from torch.utils.data import DataLoader
from loguru import logger

from .dataset import CLIPDataset
from .mvtec import load_mvtec, mvtec_classes
from .visa import load_visa, visa_classes


mean_train = [0.48145466, 0.4578275, 0.40821073]
std_train = [0.26862954, 0.26130258, 0.27577711]

load_function_dict = {
    'mvtec': load_mvtec,
    'visa': load_visa,
}

dataset_classes = {
    'mvtec': mvtec_classes,
    'visa': visa_classes,
}

def denormalization(x):
    x = (((x.transpose(1, 2, 0) * std_train) + mean_train) * 255.).astype(np.uint8)
    return x

def get_dataloader_from_args(phase, transform=None, **kwargs):

    dataset_inst = CLIPDataset(
        load_function=load_function_dict[kwargs['dataset']],
        category=kwargs['class_name'],
        phase=phase,
        k_shot=kwargs['k_shot'],
        img_resize=kwargs.get('img_resize', None),
        transform=transform,
        load_memory=kwargs.get('load_memory', False),
    )

    num_workers = int(kwargs.get('num_workers', 4))
    use_cpu = int(kwargs.get('use_cpu', 0)) == 1
    pin_memory = (not use_cpu) and kwargs.get('pin_memory', True)
    persistent_workers = num_workers > 0 and kwargs.get('persistent_workers', True)

    if phase == 'train':
        data_loader = DataLoader(dataset_inst, batch_size=kwargs['batch_size'], shuffle=True,
                                  num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    else:
        data_loader = DataLoader(dataset_inst, batch_size=kwargs['batch_size'], shuffle=False,
                                 num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)


    # debug_str = f"===> datasets: {kwargs['dataset']}, class name/len: {kwargs['class_name']}/{len(dataset_inst)}, batch size: {kwargs['batch_size']}"
    # # logger.info(debug_str)

    return data_loader, dataset_inst