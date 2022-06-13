python ./model/stylegan/prepare_data.py --out ./data/head2/lmdb/ --n_worker 4 --size 1024 ./data/head2/images/


python -m torch.distributed.launch --nproc_per_node=8 --master_port=8765 finetune_stylegan.py --iter 600 --batch 4 --ckpt ./checkpoint/stylegan2-ffhq-config-f.pt --style head2 --augment ./data/head2/lmdb/ --size 1024

python finetune_stylegan.py --iter 600 --batch 4 --ckpt /kaggle/input/zhoudualstylegan/DualStyleGAN/checkpoint/stylegan2-ffhq-config-f.pt --style head2 --augment /kaggle/input/zhoudualstylegan/DualStyleGAN/data/head2/lmdb/ --size 1024