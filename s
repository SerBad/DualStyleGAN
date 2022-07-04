python ./model/stylegan/prepare_data.py --out ./data/head2/lmdb/ --n_worker 4 --size 1024 ./data/head2/images/


python -m torch.distributed.launch --nproc_per_node=8 --master_port=8765 finetune_stylegan.py --iter 600 --batch 4 --ckpt ./checkpoint/stylegan2-ffhq-config-f.pt --style head2 --augment ./data/head2/lmdb/ --size 1024

python finetune_stylegan.py --iter 600 --batch 4 --ckpt ../input/zhoudualstylegan/DualStyleGAN/checkpoint/stylegan2-ffhq-config-f.pt --style head2 --augment ../input/zhoudualstylegan/DualStyleGAN/data/head2/lmdb/ --size 1024

python finetune_stylegan.py --iter 600 --batch 4 --ckpt /kaggle/input/zhoudualstylegan/DualStyleGAN/checkpoint/stylegan2-ffhq-config-f.pt --style head2 --augment /kaggle/input/zhoudualstylegan/DualStyleGAN/data/head2/lmdb/ --size 1024

python destylize.py --model_name fintune-000600.pt --batch 1 --iter 300 head2

python -m torch.distributed.launch --nproc_per_node=8 --master_port=8765 finetune_dualstylegan.py --iter 1500 --batch 4 --ckpt ./checkpoint/generator-pretrain.pt --style_loss 0.25 --CX_loss 0.25 --perc_loss 1 --id_loss 1 --L2_reg_loss 0.015 --augment head2
/kaggle/input/zhoudualstylegan/checkpoint
python -m torch.distributed.launch --nproc_per_node=1 --master_port=8765 pretrain_dualstylegan.py --iter 3000 --batch 4 /kaggle/input/zhoudualstylegan/DualStyleGAN/data/simpsons/lmdb/ --ckpt  /kaggle/input/zhoudualstylegan/DualStyleGAN/checkpoint/stylegan2-ffhq-config-f.pt --encoder_path /kaggle/input/zhoudualstylegan/DualStyleGAN/checkpoint/encoder.pt


!python finetune_dualstylegan.py --iter 1500 --batch 4 --ckpt /kaggle/input/zhoudualstylegan/DualStyleGAN/checkpoint/simpsons/generator-pretrain.pt --style_loss 0.25 --CX_loss 0.25 --perc_loss 0.8 --id_loss 1 --L2_reg_loss 0.015 --encoder_path /kaggle/input/zhoudualstylegan/DualStyleGAN/checkpoint/encoder.pt  --image_path /kaggle/input/zhoudualstylegan/DualStyleGAN/data/simpsons/images/train --lmdb_path /kaggle/input/zhoudualstylegan/DualStyleGAN/data/simpsons/lmdb --identity_path /kaggle/input/zhoudualstylegan/DualStyleGAN/checkpoint/model_ir_se50.pth --exstyle_path /kaggle/input/zhoudualstylegan/DualStyleGAN/checkpoint/simpsons/exstyle_code.npy --instyle_path /kaggle/input/zhoudualstylegan/DualStyleGAN/checkpoint/simpsons/instyle_code.npy --augment simpsons




python ./model/stylegan/prepare_data.py --out ./data/caricature/lmdb/ --n_worker 4 --size 1024 ./data/caricature/images/
python finetune_stylegan.py --iter 600 --batch 4 --ckpt ./checkpoint/stylegan2-ffhq-config-f.pt --style caricature --augment ./data/caricature/lmdb/
python -m torch.distributed.launch --nproc_per_node=8 --master_port=8765 finetune_stylegan.py --iter 600 --batch 4 --ckpt ./checkpoint/stylegan2-ffhq-config-f.pt --style caricature --augment ./data/caricature/lmdb/
python finetune_stylegan.py --iter 600 --batch 4 --ckpt /kaggle/input/zhoudualstylegan/DualStyleGAN/checkpoint/stylegan2-ffhq-config-f.pt --style head2 --augment /kaggle/input/zhoudualstylegan/DualStyleGAN/data/caricature/lmdb/ --size 1024
