python test.py \
--resume './outputs/train_bone_msunet/checkpoints/model_best.pt' \
--experiment_name 'test_bone_msunet' \
--model_type 'model_bone' \
--data_root './example_data/' \
--net_G 'msunet' \
--net_D 'patchGAN' \
--gpu_ids 0

