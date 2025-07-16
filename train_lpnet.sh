python setup.py develop --no_cuda_ext
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4354 basicsr/train_ir.py -opt options/train/LPNet_train.yml --launcher pytorch

