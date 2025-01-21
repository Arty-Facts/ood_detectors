import ood_detectors.vision as vision
import ood_detectors.ops_utils as ops_utils
import torch
import random
import pathlib
import pickle


import multiprocessing as mp

def encode_dataset(encoder_name, dataset_name, aug=False, device='cpu'):
    data_root = '/mnt/data/arty/data'
    image_list_root = '/mnt/data/arty/data/benchmark_imglist'
    features = pathlib.Path("/mnt/data/arty/data/features_ood_2025")
    features.mkdir(exist_ok=True)
    encoder_path = features / encoder_name
    encoder_path.mkdir(exist_ok=True)
    dataset_name_path = encoder_path / dataset_name
    dataset_name_path.mkdir(exist_ok=True)
    encoder = vision.get_encoder(encoder_name)
    encoder.eval()
    if aug:
        datasets = vision.get_datasets(dataset_name, data_root=data_root, image_list_root=image_list_root, transform=encoder.transform, train_transform=encoder.train_transform)
    else:
        datasets = vision.get_datasets(dataset_name, data_root=data_root, image_list_root=image_list_root, transform=encoder.transform)
    for type_name, datasets in datasets.items():
        if type_name == 'csid':
            continue
        type_name_path = dataset_name_path / type_name
        type_name_path.mkdir(exist_ok=True)
        for name, dataset in datasets.items():
            name_path = type_name_path / name
            name_path.mkdir(exist_ok=True)
            features_path = name_path / 'feature_data.pkl'
            if features_path.exists():
                print('Skipping', encoder_name, dataset_name, name, type_name)
                continue
            if aug:
                features_vectors = []
                for _ in range(10):
                    features_vectors.append(vision.extract_features(encoder, dataset, batch_size=64, num_workers=2, device=device))
                features_vectors = torch.cat(features_vectors, dim=0)
            else:        
                features_vectors = vision.extract_features(encoder, dataset, batch_size=64, num_workers=2, device=device)
            
            print(encoder_name, dataset_name, name, type_name, features_vectors.shape)
            data = {
                'features': features_vectors,
                'encoder': encoder_name,
                'target_dataset': dataset_name,
                'dataset': name,
                'type': type_name
            }
            with open(features_path, 'wb') as f:
                pickle.dump(data, f)
def main():

    # encoders = ['repvgg', 'resnet50d', 'swin', 'deit', 'dino', 'dinov2', 'vit', 'clip', , 'swin_t']
    encoders = ['repvgg', 'resnet50d', 'swin', 'deit', 'bit', 'dino', 'dinov2']
    # datasets = ['imagenet', 'imagenet200', 'cifar10', 'cifar100']
    datasets = ['imagenet']
    gpu_nodes = [0, 1, 2, 3]
    aug = True

    jobs = []
    for encoder_name in encoders:
        for dataset_name in datasets:
            jobs.append((encoder_name, dataset_name, aug))
        
    # for encoder_name, dataset_name in zip(['resnet18_32x32_cifar10_open_ood', 
    #                             'resnet18_32x32_cifar100_open_ood', 
    #                             'resnet18_224x224_imagenet200_open_ood', 
    #                             'resnet50_224x224_imagenet_open_ood'],
    #                             ['cifar10', 'cifar100', 'imagenet200', 'imagenet']):
    #     jobs.append((encoder_name, dataset_name, aug))

    print(len(jobs))
    random.shuffle(jobs)


    ops_utils.parallelize(encode_dataset, jobs, gpu_nodes, verbose=True, timeout=60*60*24)

if __name__ == '__main__':
    mp.freeze_support()
    main()