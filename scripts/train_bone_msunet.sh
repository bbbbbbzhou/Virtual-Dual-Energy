python train.py \
--experiment_name 'train_bone_msunet' \
--model_type 'model_bone' \
--dataset 'DE' \
--data_root './example_data/' \
--net_G 'msunet' \
--net_D 'patchGAN' \
--wr_recon 50 \
--batch_size 2 \
--lr 1e-4 \
--AUG \
--eval_epochs 8 \
--save_epochs 8 \
--snapshot_epochs 8 \
--gpu_ids 3

