# MultiCBR
Pytorch implementation for "MultiCBR: Multi-view Contrastive Learning for Bundle Recommendation"

### Environment

- OS: Ubuntu 18.04 or higher version
- python == 3.7.11 or above
- supported(tested) CUDA versions: 10.2
- Pytorch == 1.9.0 or above

### Run the code
To train MultiCBR on dataset NetEase with GPU 0, simply run:

    python train.py -g 0 -m MultiCBR -d NetEase
You can indicate GPU id or dataset with cmd line arguments, and the hyper-parameters are recorded in config.yaml. 
