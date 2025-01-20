import timm
import torch
import torchvision
import pathlib, requests, pathlib, gdown, zipfile
from torchvision.datasets import ImageFolder
from ood_detectors.list_dataset import ImageFilelist
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, ResNet, Bottleneck

def download(url, filename):
    data = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(data.content)

class TimmModel(torch.nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.name = model_name
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        data_config = timm.data.resolve_model_data_config(self.model)   
        self.transform = timm.data.create_transform(**data_config, is_training=True, no_aug=True)
        self.train_transform = timm.data.create_transform(**data_config, is_training=True, no_aug=False)


    def forward(self, x):
        x = self.model(x)
        return x
    
    def features(self, x):
        x = self.model(x)
        return x



class Dino_ViT_B_16(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'dino_vit_b_16'
        self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def forward(self, x):
        x = self.model(x)
        return x
    
    def features(self, x):
        x = self.model(x)
        return x

class DinoV2_ViT_B_14(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'dinov2_vit_b_14'
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def forward(self, x):
        x = self.model(x)
        return x
    
    def features(self, x):
        x = self.model(x)
        return x
 
class ViT_P16_21k(torch.nn.Module):
    def __init__(self, model_dir_path='checkpoints'):
        from mmpretrain.apis import init_model
        import mmengine
        super().__init__()
        self.name = 'vit_p_16_21k'
        model_dir = pathlib.Path(model_dir_path)
        weights = model_dir/'vit-base-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-98e8652b.pth'
        if not model_dir.exists():
            model_dir.mkdir(exist_ok=True)
        if not weights.exists():
            url = 'https://download.openmmlab.com/mmclassification/v0/vit/finetune/vit-base-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-98e8652b.pth'
            download(url, weights)
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((384, 384)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(384),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        cfg = mmengine.Config(cfg_dict=dict(model=dict(
            type='ImageClassifier',
            backbone=dict(
                type='VisionTransformer',
                arch='b',
                img_size=384,
                patch_size=16,
                drop_rate=0.1,
                init_cfg=[
                    dict(
                        type='Kaiming',
                        layer='Conv2d',
                        mode='fan_in',
                        nonlinearity='linear')
                ]),
            neck=None,
            head=dict(
                type='VisionTransformerClsHead',
                num_classes=1000,
                in_channels=768,
                loss=dict(
                    type='LabelSmoothLoss', label_smooth_val=0.1,
                    mode='classy_vision'),
            )))
        )

        self.model = init_model(cfg, str(weights), 0)
        self.device = 'cpu'

    def to(self, device):
        self.model.to(device)
        self.device = device
        return self

    def forward(self, x):
        x = self.model(x)
        return x
    
    def features(self, x):
        x = self.model.backbone(x)[0]
        return x
    
class ClipVisionModel(torch.nn.Module):
    def __init__(self):
        import clip
        super().__init__()
        self.name = 'CLIP'
        self.model, self.transform = clip.load("ViT-B/32", device='cpu')
        self.train_transform = self.transform
        self.device = 'cpu'

    def to(self, device):
        self.model.visual.to(device) # Move the visual model to the specified device we don't need the text model
        self.device = device
        return self

    def forward(self, x):
        x = self.model.encode_image(x)
        return x

    def features(self, x):
        x = self.model.encode_image(x)
        return x

class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18_32x32(nn.Module):
    def __init__(self, block=_BasicBlock, num_blocks=None, num_classes=10):
        super().__init__()
        if num_blocks is None:
            num_blocks = [2, 2, 2, 2]
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3,
                               64,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # self.avgpool = nn.AvgPool2d(4)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.feature_size = 512 * block.expansion

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def features(self, x):
        feature1 = F.relu(self.bn1(self.conv1(x)))
        feature2 = self.layer1(feature1)
        feature3 = self.layer2(feature2)
        feature4 = self.layer3(feature3)
        feature5 = self.layer4(feature4)
        feature5 = self.avgpool(feature5)
        feature = feature5.view(feature5.size(0), -1)
        return feature

    def forward(self, x):
        return self.fc(self.features(x))
    
class ResNet18_224x224(ResNet):
    def __init__(self,
                 block=BasicBlock,
                 layers=[2, 2, 2, 2],
                 num_classes=1000):
        super().__init__(block=block,
                                               layers=layers,
                                               num_classes=num_classes)
        self.feature_size = 512

    def features(self, x):
        feature1 = self.relu(self.bn1(self.conv1(x)))
        feature1 = self.maxpool(feature1)
        feature2 = self.layer1(feature1)
        feature3 = self.layer2(feature2)
        feature4 = self.layer3(feature3)
        feature5 = self.layer4(feature4)
        feature5 = self.avgpool(feature5)
        feature = feature5.view(feature5.size(0), -1)
        return feature
        
    def forward(self, x):
        return self.fc(self.features(x))
    
class ResNet50(ResNet):
    def __init__(self,
                 block=Bottleneck,
                 layers=[3, 4, 6, 3],
                 num_classes=1000):
        super(ResNet50, self).__init__(block=block,
                                       layers=layers,
                                       num_classes=num_classes)
        self.feature_size = 2048

    def features(self, x):
        feature1 = self.relu(self.bn1(self.conv1(x)))
        feature1 = self.maxpool(feature1)
        feature2 = self.layer1(feature1)
        feature3 = self.layer2(feature2)
        feature4 = self.layer3(feature3)
        feature5 = self.layer4(feature4)
        feature5 = self.avgpool(feature5)
        feature = feature5.view(feature5.size(0), -1)
        return feature
    
    def forward(self, x):
        return self.fc(self.features(x))

class Swin_T(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "swin_t"
        self.model = torchvision.models.swin_t(weights=torchvision.models.Swin_T_Weights.IMAGENET1K_V1)
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def features(self, x):
        x = self.model.features(x)
        x = self.model.norm(x)
        x = self.model.permute(x)
        x = self.model.avgpool(x)
        x = self.model.flatten(x)
        return x

    def forward(self, x):
        x = self.model.features(x)
        return self.model.head(x)


normalization_dict = {
    'cifar10': [[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]],
    'cifar100': [[0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]],
    'imagenet': [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
}

download_id_open_ood = {
    'cifar10_res18_v1.5': '1byGeYxM_PlLjT72wZsMQvP6popJeWBgt',
    'cifar100_res18_v1.5': '1s-1oNrRtmA0pGefxXJOUVRYpaoAML0C-',
    'imagenet200_res18_v1.5': '1ddVmwc8zmzSjdLUO84EuV4Gz1c7vhIAs',
    'imagenet_res50_v1.5': '15PdDMNRfnJ7f2oxW6lI-Ge4QJJH3Z0Fy',
    'datalist': '1XKzBdWCqg3vPoj-D32YixJyJJ0hL63gP',
    'usps': '1KhbWhlFlpFjEIb4wpvW0s9jmXXsHonVl',
    'cifar100': '1PGKheHUsf29leJPPGuXqzLBMwl8qMF8_',
    'cifar10': '1Co32RiiWe16lTaiOU6JMMnyUYS41IlO1',
    'cifar10c': '170DU_ficWWmbh6O2wqELxK9jxRiGhlJH',
    'cinic10': '190gdcfbvSGbrRK6ZVlJgg5BqqED6H_nn',
    'svhn': '1DQfc11HOtB1nEwqS4pWUFp8vtQ3DczvI',
    'fashionmnist': '1nVObxjUBmVpZ6M0PPlcspsMMYHidUMfa',
    'cifar100c': '1MnETiQh9RTxJin2EHeSoIAJA28FRonHx',
    'mnist': '1CCHAGWqA1KJTFFswuF9cbhmB-j98Y1Sb',
    'tin': '1PZ-ixyx52U989IKsMA2OT-24fToTrelC',
    'texture': '1OSz1m3hHfVWbRdmMwKbUzoU8Hg9UKcam',
    'notmnist': '16ueghlyzunbksnc_ccPgEAloRW9pKO-K',
    'places365': '1Ec-LRSTf6u5vEctKX9vRp9OA6tqnJ0Ay',
    'places': '1fZ8TbPC4JGqUCm-VtvrmkYxqRNp2PoB3',
    'sun': '1ISK0STxWzWmg-_uUr4RQ8GSLFW7TZiKp',
    'species_sub': '1-JCxDx__iFMExkYRMylnGJYTPvyuX6aq',
    'imagenet_1k': '1i1ipLDFARR-JZ9argXd2-0a6DXwVhXEj',
    'ssb_hard': '1PzkA-WGG8Z18h0ooL_pDdz9cO-DCIouE',
    'ninco': '1Z82cmvIB0eghTehxOGP5VTdLt7OD3nk6',
    'imagenet_v2': '1akg2IiE22HcbvTBpwXQoD7tgfPCdkoho',
    'imagenet_r': '1EzjMN2gq-bVV7lg-MEAdeuBuz-7jbGYU',
    'imagenet_c': '1JeXL9YH4BO8gCJ631c5BHbaSsl-lekHt',
    'imagenet_o': '1S9cFV7fGvJCcka220-pIO9JPZL1p1V8w',
    'openimage_o': '1VUFXnB_z70uHfdgJG2E_pjYOcEgqM7tE',
    'inaturalist': '1zfLfMvoUD0CUlKNnkk7LgxZZBnTBipdj',
    'actmed': '1tibxL_wt6b3BjliPaQ2qjH54Wo4ZXWYb',
    'ct': '1k5OYN4inaGgivJBJ5L8pHlopQSVnhQ36',
    'hannover': '1NmqBDlcA1dZQKOvgcILG0U1Tm6RP0s2N',
    'xraybone': '1ZzO3y1-V_IeksJXEvEfBYKRoQLLvPYe9',
    'bimcv': '1nAA45V6e0s5FAq2BJsj9QH5omoihb7MZ',
}

data_list_open_ood = [
    'usps',
    'cifar100',
    'cifar10',
    'cifar10c',
    'cinic10',
    'svhn',
    'fashionmnist',
    'cifar100c',
    'mnist',
    'tin',
    'texture',
    'notmnist',
    'places365',
    'places',
    'sun',
    'species_sub',
    'imagenet_1k',
    'ssb_hard',
    'ninco',
    'imagenet_v2',
    'imagenet_r',
    'imagenet_c',
    'imagenet_o',
    'openimage_o',
    'inaturalist',
    'actmed',
    'ct',
    'hannover',
    'xraybone',
    'bimcv',
]

AVAILABLE_ENCODERS = ['repvgg', 'resnet50d', 'swin', 'bit', 'deit', 'dino', 'dinov2', 'vit', 'clip', 'swin_t', 'resnet18_32x32_cifar10_open_ood', 'resnet18_32x32_cifar100_open_ood', 'resnet18_224x224_imagenet200_open_ood', 'resnet50_224x224_imagenet_open_ood']

def download_and_extract(hash, checkpoint_path):
    gdown.download(id=hash, output=str(checkpoint_path / f"{hash}.zip") )
    with zipfile.ZipFile(checkpoint_path/f"{hash}.zip", 'r') as zip_ref:
        zip_ref.extractall(checkpoint_path)
    (checkpoint_path/f"{hash}.zip").unlink()


def get_encoder(name):
    name = name.lower()
    if name in ['repvgg', 'repvgg_b3']:
        return TimmModel('repvgg_b3')
    elif name in ['resnet50d', 'res50d', 'resnet50']:
        return TimmModel('resnet50d')
    elif name in ['swin', 'swin_base_patch4_window7_224']:
        return TimmModel('swin_base_patch4_window7_224')
    elif name in ['deit', 'deit_base_patch16_224']:
        return TimmModel('deit_base_patch16_224')
    elif name in ['bit', 'resnetv2_101x1_bit.goog_in21k_ft_in1k']:
        return TimmModel('resnetv2_101x1_bit.goog_in21k_ft_in1k')
    elif name in ['dino', 'dino_vit_b_16']:
        return Dino_ViT_B_16()
    elif name in ['dinov2', 'dinov2_vit_b_14']:
        return DinoV2_ViT_B_14()
    elif name in ['vit','vit_p16']:
        return ViT_P16_21k()
    elif name in ['swin_t']:
        return Swin_T()
    elif name in ['clip']:
        return ClipVisionModel()
    elif name in ['resnet18_32x32_cifar10_open_ood']:
        model = ResNet18_32x32()
        checkpoint = pathlib.Path('checkpoints/')
        checkpoint.mkdir(exist_ok=True)
        model_path = checkpoint/'cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best_epoch96_acc0.9470.ckpt'
        if not model_path.exists():
            download_and_extract(download_id_open_ood['cifar10_res18_v1.5'], checkpoint)
        model.load_state_dict(torch.load(model_path))
        model.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(*normalization_dict['cifar10'])
        ])

        model.train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(32),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(*normalization_dict['cifar10'])
        ])
        model.name = 'resnet18_32x32_cifar10_open_ood'
        return model
    elif name in ['resnet18_32x32_cifar100_open_ood']:
        model = ResNet18_32x32(num_classes=100)
        checkpoint = pathlib.Path('checkpoints/')
        checkpoint.mkdir(exist_ok=True)
        model_path = checkpoint/'cifar100_resnet18_32x32_base_e100_lr0.1_default/s0/best_epoch99_acc0.7810.ckpt'
        if not model_path.exists():
            download_and_extract(download_id_open_ood['cifar100_res18_v1.5'], checkpoint)
        model.load_state_dict(torch.load(model_path))
        model.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(*normalization_dict['cifar100'])
        ])

        model.train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(32),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(*normalization_dict['cifar100'])
        ])
        model.name = 'resnet18_32x32_cifar100_open_ood'
        return model
    elif name in ['resnet18_224x224_imagenet200_open_ood']:
        model = ResNet18_224x224(num_classes=200)
        checkpoint = pathlib.Path('checkpoints/')
        checkpoint.mkdir(exist_ok=True)
        model_path = checkpoint/'imagenet200_resnet18_224x224_base_e90_lr0.1_default/s0/best_epoch89_acc0.8500.ckpt'
        if not model_path.exists():
            download_and_extract(download_id_open_ood['imagenet200_res18_v1.5'], checkpoint)
        model.load_state_dict(torch.load(model_path))
        model.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(*normalization_dict['imagenet'])
        ])

        model.train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(*normalization_dict['imagenet'])
        ])

        model.name = 'resnet18_224x224_imagenet200_open_ood'
        return model
    elif name in ['resnet50_224x224_imagenet_open_ood']:
        model = ResNet50(num_classes=1000)
        checkpoint = pathlib.Path('checkpoints/')
        checkpoint.mkdir(exist_ok=True)
        model_path = checkpoint/'imagenet_resnet50_base_e30_lr0.001_randaugment-2-9/s0/best_epoch27_acc0.7668.ckpt'
        if not model_path.exists():
            download_and_extract(download_id_open_ood['imagenet_res50_v1.5'], checkpoint)
        model.load_state_dict(torch.load(model_path))
        model.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(*normalization_dict['imagenet'])
        ])

        model.train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(*normalization_dict['imagenet'])
        ])

        model.name = 'resnet50_224x224_imagenet_open_ood'
        return model
    else:
        raise ValueError(f"Encoder {name} not available. Available encoders: {AVAILABLE_ENCODERS}")
    
def get_open_ood_datalist(checkpoint_path=pathlib.Path('checkpoints/')):
    if not (checkpoint_path/'benchmark_imglist').exists():
        download_and_extract(download_id_open_ood['datalist'], checkpoint_path)

def get_open_ood_datasets(checkpoint_path=pathlib.Path('checkpoints/')):
    for dataset in data_list_open_ood:
        if not (checkpoint_path/dataset).exists():
            download_and_extract(download_id_open_ood[dataset], checkpoint_path)

def build_dataset(data_root, transform, img_list=None):
    if img_list is not None:
        dataset = ImageFilelist(data_root, img_list, transform)
    else:
        dataset = ImageFolder(data_root, transform, allow_empty=True)
    return dataset

def imagenet_ood_datasets(data_root, transforms, train_transform, image_list_root):
    datasets = {}
    datasets['id'] = {}
    datasets['nearood'] = {}
    datasets['farood'] = {}
    datasets['csid'] = {}
    datasets['id']['train'] = build_dataset(data_root, train_transform, img_list=f'{image_list_root}/imagenet/train_imagenet.txt')
    datasets['id']['val'] = build_dataset(data_root, transforms, img_list=f'{image_list_root}/imagenet/val_imagenet.txt')
    datasets['id']['test'] = build_dataset(data_root, transforms, img_list=f'{image_list_root}/imagenet/test_imagenet.txt')
    datasets['nearood']['ssb_hard'] = build_dataset(data_root, transforms, img_list=f'{image_list_root}/imagenet/test_ssb_hard.txt')
    datasets['nearood']['ninco'] = build_dataset(data_root, transforms, img_list=f'{image_list_root}/imagenet/test_ninco.txt')
    datasets['nearood']['imagenet-o'] = build_dataset(f'{data_root}/imagenet-o/', transforms, img_list="all")
    datasets['farood']['textures'] = build_dataset(data_root, transforms, img_list=f'{image_list_root}/imagenet/test_textures.txt')
    datasets['farood']['inaturalist'] = build_dataset(data_root, transforms, img_list=f'{image_list_root}/imagenet/test_inaturalist.txt')
    datasets['farood']['openimageo'] = build_dataset(data_root, transforms, img_list=f'{image_list_root}/imagenet/test_openimage_o.txt')
    datasets['csid']['imagenetv2'] = build_dataset(data_root, transforms, img_list=f'{image_list_root}/imagenet/test_imagenet_v2.txt')
    datasets['csid']['imagenetc'] = build_dataset(data_root, transforms, img_list=f'{image_list_root}/imagenet/test_imagenet_c.txt')
    datasets['csid']['imagenetr'] = build_dataset(data_root, transforms, img_list=f'{image_list_root}/imagenet/test_imagenet_r.txt')
    return datasets
    

def imagenet200_ood_datasets(data_root, transforms, train_transform, image_list_root):
    datasets = {}
    datasets['id'] = {}
    datasets['nearood'] = {}
    datasets['farood'] = {}
    datasets['csid'] = {}
    datasets['id']['train'] = build_dataset(data_root, train_transform, img_list=f'{image_list_root}/imagenet200/train_imagenet200.txt')
    datasets['id']['val'] = build_dataset(data_root, transforms, img_list=f'{image_list_root}/imagenet200/val_imagenet200.txt')
    datasets['id']['test'] = build_dataset(data_root, transforms, img_list=f'{image_list_root}/imagenet200/test_imagenet200.txt')
    datasets['nearood']['ssb_hard'] = build_dataset(data_root, transforms, img_list=f'{image_list_root}/imagenet/test_ssb_hard.txt')
    datasets['nearood']['ninco'] = build_dataset(data_root, transforms, img_list=f'{image_list_root}/imagenet/test_ninco.txt')
    datasets['nearood']['imagenet-o'] = build_dataset(f'{data_root}/imagenet-o/', transforms, img_list="all")
    datasets['farood']['textures'] = build_dataset(data_root, transforms, img_list=f'{image_list_root}/imagenet/test_textures.txt')
    datasets['farood']['inaturalist'] = build_dataset(data_root, transforms, img_list=f'{image_list_root}/imagenet/test_inaturalist.txt')
    datasets['farood']['openimageo'] = build_dataset(data_root, transforms, img_list=f'{image_list_root}/imagenet/test_openimage_o.txt')
    datasets['csid']['imagenetv2'] = build_dataset(data_root, transforms, img_list=f'{image_list_root}/imagenet/test_imagenet_v2.txt')
    datasets['csid']['imagenetc'] = build_dataset(data_root, transforms, img_list=f'{image_list_root}/imagenet/test_imagenet_c.txt')
    datasets['csid']['imagenetr'] = build_dataset(data_root, transforms, img_list=f'{image_list_root}/imagenet/test_imagenet_r.txt')
    return datasets


def cifar10_ood_datasets(data_root, transforms, train_transform, image_list_root):
    datasets = {}
    datasets['id'] = {}
    datasets['nearood'] = {}
    datasets['farood'] = {}
    datasets['csid'] = {}
    datasets['id']['train'] = build_dataset(data_root, train_transform, img_list=f'{image_list_root}/cifar10/train_cifar10.txt')
    datasets['id']['val'] = build_dataset(data_root, transforms, img_list=f'{image_list_root}/cifar10/val_cifar10.txt')
    datasets['id']['test'] = build_dataset(data_root, transforms, img_list=f'{image_list_root}/cifar10/test_cifar10.txt')
    datasets['nearood']['cifar100'] = build_dataset(data_root, transforms, img_list=f'{image_list_root}/cifar10/test_cifar100.txt')
    datasets['nearood']['tin'] = build_dataset(data_root, transforms, img_list=f'{image_list_root}/cifar10/test_tin.txt')
    datasets['farood']['mnist'] = build_dataset(data_root, transforms, img_list=f'{image_list_root}/cifar10/test_mnist.txt')
    datasets['farood']['svhn'] = build_dataset(data_root, transforms, img_list=f'{image_list_root}/cifar10/test_svhn.txt')
    datasets['farood']['texture'] = build_dataset(data_root, transforms, img_list=f'{image_list_root}/cifar10/test_texture.txt')
    datasets['farood']['place365'] = build_dataset(data_root, transforms, img_list=f'{image_list_root}/cifar10/test_places365.txt')
    datasets['csid']['cinic10'] = build_dataset(f"{data_root}/cinic10/val", transforms, img_list="all")
    return datasets

def cifar100_ood_datasets(data_root, transforms, train_transform, image_list_root):
    datasets = {}
    datasets['id'] = {}
    datasets['nearood'] = {}
    datasets['farood'] = {}
    datasets['csid'] = {}
    datasets['id']['train'] = build_dataset(data_root, train_transform, img_list=f'{image_list_root}/cifar100/train_cifar100.txt')
    datasets['id']['val'] = build_dataset(data_root, transforms, img_list=f'{image_list_root}/cifar100/val_cifar100.txt')
    datasets['id']['test'] = build_dataset(data_root, transforms, img_list=f'{image_list_root}/cifar100/test_cifar100.txt')
    datasets['nearood']['cifar10'] = build_dataset(data_root, transforms, img_list=f'{image_list_root}/cifar100/test_cifar10.txt')
    datasets['nearood']['tin'] = build_dataset(data_root, transforms, img_list=f'{image_list_root}/cifar100/test_tin.txt')
    datasets['farood']['mnist'] = build_dataset(data_root, transforms, img_list=f'{image_list_root}/cifar100/test_mnist.txt')
    datasets['farood']['svhn'] = build_dataset(data_root, transforms, img_list=f'{image_list_root}/cifar100/test_svhn.txt')
    datasets['farood']['texture'] = build_dataset(data_root, transforms, img_list=f'{image_list_root}/cifar100/test_texture.txt')
    datasets['farood']['places365'] = build_dataset(data_root, transforms, img_list=f'{image_list_root}/cifar100/test_places365.txt')
    datasets['csid']['cifar100c'] = build_dataset(f"{data_root}/cifar100c/", transforms, img_list="all")
    return datasets

def covid_ood_datasets(data_root, transforms, train_transform, image_list_root):
    datasets = {}
    datasets['id'] = {}
    datasets['nearood'] = {}
    datasets['farood'] = {}
    datasets['csid'] = {}
    datasets['id']['train'] = build_dataset(data_root, train_transform, img_list=f'{image_list_root}/covid/train_bimcv.txt')
    datasets['id']['val'] = build_dataset(data_root, transforms, img_list=f'{image_list_root}/covid/val_bimcv.txt')
    datasets['id']['test'] = build_dataset(data_root, transforms, img_list=f'{image_list_root}/covid/test_bimcv.txt')
    datasets['nearood']['ct'] = build_dataset(data_root, transforms, img_list=f'{image_list_root}/covid/test_ct.txt')
    datasets['nearood']['xraybone'] = build_dataset(data_root, transforms, img_list=f'{image_list_root}/covid/test_xraybone.txt')
    datasets['farood']['mnist'] = build_dataset(data_root, transforms, img_list=f'{image_list_root}/covid/test_mnist.txt')
    datasets['farood']['cifar10'] = build_dataset(data_root, transforms, img_list=f'{image_list_root}/covid/test_cifar10.txt')
    datasets['farood']['texture'] = build_dataset(data_root, transforms, img_list=f'{image_list_root}/covid/test_texture.txt')
    datasets['farood']['tin'] = build_dataset(f"{data_root}/tin/val/", transforms, img_list="all")
    datasets['csid']['actmed'] = build_dataset(data_root, transforms, img_list=f'{image_list_root}/covid/test_actmed.txt')
    datasets['csid']['hannover'] = build_dataset(data_root, transforms, img_list=f'{image_list_root}/covid/test_hannover.txt')
    return datasets

def mnist_ood_datasets(data_root, transforms, train_transform, image_list_root):
    datasets = {}
    datasets['id'] = {}
    datasets['nearood'] = {}
    datasets['farood'] = {}
    datasets['csid'] = {}
    datasets['id']['train'] = build_dataset(data_root, train_transform, img_list=f'{image_list_root}/mnist/train_mnist.txt')
    datasets['id']['val'] = build_dataset(data_root, transforms, img_list=f'{image_list_root}/mnist/val_mnist.txt')
    datasets['id']['test'] = build_dataset(data_root, transforms, img_list=f'{image_list_root}/mnist/test_mnist.txt')
    datasets['nearood']['notmnist'] = build_dataset(data_root, transforms, img_list=f'{image_list_root}/mnist/test_notmnist.txt')
    datasets['nearood']['fashionmnist'] = build_dataset(data_root, transforms, img_list=f'{image_list_root}/mnist/test_fashionmnist.txt')
    datasets['farood']['texture'] = build_dataset(data_root, transforms, img_list=f'{image_list_root}/mnist/test_texture.txt')
    datasets['farood']['cifar10'] = build_dataset(data_root, transforms, img_list=f'{image_list_root}/mnist/test_cifar10.txt')
    datasets['farood']['tin'] = build_dataset(data_root, transforms, img_list=f'{image_list_root}/mnist/test_tin.txt')
    datasets['farood']['places365'] = build_dataset(data_root, transforms, img_list=f'{image_list_root}/mnist/test_places365.txt')
    datasets['csid']['svhn'] = build_dataset(data_root, transforms, img_list=f'{image_list_root}/mnist/test_svhn.txt')
    return datasets

AVAILABLE_DATASETS = ['imagenet', 'imagenet200', 'cifar10', 'cifar100', 'covid', 'mnist']

def get_datasets(name, data_root, image_list_root, transform, train_transform=None):
    if train_transform is not None:
        train_transform = transform
    if name == 'imagenet':
        return imagenet_ood_datasets(data_root, transform, train_transform, image_list_root)
    elif name == 'imagenet200':
        return imagenet200_ood_datasets(data_root, transform, train_transform, image_list_root)
    elif name == 'cifar10':
        return cifar10_ood_datasets(data_root, transform, train_transform, image_list_root)
    elif name == 'cifar100':
        return cifar100_ood_datasets(data_root, transform, train_transform, image_list_root)
    elif name == 'covid':
        return covid_ood_datasets(data_root, transform, train_transform, image_list_root)
    elif name == 'mnist':
        return mnist_ood_datasets(data_root, transform, train_transform, image_list_root)
    else:
        raise ValueError(f"Dataset {name} not available. Available datasets: {AVAILABLE_DATASETS}")
    
def extract_features(encoder, dataset, batch_size=64, num_workers=8, device='cuda'):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    features = []
    encoder.to(device)
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            feature = encoder.features(x).cpu()
            features.append(feature)
    features = torch.cat(features)
    return features