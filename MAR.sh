\python setup.py develop --no_cuda_ext
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4329 basicsr/train_ir.py -opt options/train/MAR_train.yml --launcher pytorch
