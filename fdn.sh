\python setup.py develop --no_cuda_ext
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4329 basicsr/train_ir.py -opt options/train/FDN.yml --launcher pytorch
