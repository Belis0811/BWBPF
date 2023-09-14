# BWBPF: Block-Wise BP free learning
Paper is implemented with official pytorch
>[Unlocking Deep Learning: A BP-Free Approach for Parallel Block-Wise Training of Neural Networks]
>Anzhe Cheng, Zhenkun Wang, Chenzhong Yin etal.
![alt text](misc/overview_distributednn.png?raw=true "Weight updates of BWBPF.")
To overcome the drawbacks of BP, particularly the issue of backward locking, we propose the BWBPF learning algorithm which eliminates BP for the global prediction loss and instead computes the local prediction loss.

## Requirements
`Python >= 3.8, PyTorch >= 1.6.0, torchvision >= 0.7.0`

## Training Models

#### 1. original
first direct to 'original' folder by
```
cd original
```

run main_vgg.py to train cifar10 with VGG19
```
python main_vgg.py
```

run main_cifar10.py to train cifar10 with ResNet50/101/152 by changing `net = ResNet.ResNet50` to `net = ResNet.ResNet101` or `net = ResNet.ResNet152`
```
python main_cifar10.py
```

run main_tinyImage.py to train tinyImageNet with ResNet. Change `net = ResNet.ResNet50` to  `net = ResNet.ResNet101` or `net = ResNet.ResNet152` to test different model
```
python main_tinyImage.py
```
#### 2. BWBPF
direct to 'distributed' folder
```
cd distributed
```

train VGG with 4 outputs using cifar10
```
python main_vgg.py
```

run main_cifar10.py to train cifar10 with 4 outputs ResNet50/101/152 by changing `net = ResNet_4out.ResNet50` to `net = ResNet_4out.ResNet101` or `net = ResNet_4out.ResNet152`
```
python main_cifar10.py
```

train 8 outputs and 12 outputs by running
```
python Main_8out.py
```
and
```
python Main_12out.py
```
