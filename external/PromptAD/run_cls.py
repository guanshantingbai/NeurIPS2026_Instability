import os
from datasets import dataset_classes
from multiprocessing import Pool

# 限制 CPU 线程数，避免多进程并行时 CPU 超载
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

# Global output path for all training runs.
ROOT_DIR = "./result_round1"
EVAL_FREQ = 2

if __name__ == '__main__':

    pool = Pool(processes=1)

#    datasets = ['mvtec', 'visa']
#    shots = [1, 2, 4]
    datasets = ['mvtec']
    shots = [1, 2, 4]
    # task = 'train'
    task = 'test'
    vis = False


    if task == 'train':
        for shot in shots:
            for dataset in datasets:
                classes = dataset_classes[dataset]
                for cls in classes[:]:
                    sh_method = f'python train_cls.py ' \
                                f'--dataset {dataset} ' \
                                f'--k-shot {shot} ' \
                                f'--class_name {cls} ' \
                                f'--eval-freq {EVAL_FREQ} ' \
                                f'--root-dir {ROOT_DIR} '
                    print(sh_method)
                    pool.apply_async(os.system, (sh_method,))

    elif task == 'test':
        for shot in shots:
            for dataset in datasets:
                classes = dataset_classes[dataset]
                for cls in classes[:]:
                    sh_method = f'python test_cls.py ' \
                                f'--dataset {dataset} ' \
                                f'--k-shot {shot} ' \
                                f'--class_name {cls} ' \
                                f'--root-dir {ROOT_DIR} ' \
                                f'--vis {vis} '
                    print(sh_method)
                    pool.apply_async(os.system, (sh_method,))

    pool.close()
    pool.join()

