import os
from datasets import dataset_classes
from multiprocessing import Pool

# 限制 CPU 线程数，避免多进程并行时 CPU 超载
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

if __name__ == '__main__':

    pool = Pool(processes=1)

    datasets = ['mvtec', 'visa']
    shots = [1, 2, 4]

    for shot in shots:
        for dataset in datasets:
            classes = dataset_classes[dataset]
            for cls in classes[:]:
                sh_method = f'python train_seg.py ' \
                            f'--dataset {dataset} ' \
                            f'--k-shot {shot} ' \
                            f'--class_name {cls} ' \

                print(sh_method)
                pool.apply_async(os.system, (sh_method,))

    pool.close()
    pool.join()




