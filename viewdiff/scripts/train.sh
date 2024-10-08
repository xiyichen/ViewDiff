#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

export CO3DV2_DATASET_ROOT=$1

# accelerate launch --num_processes=2 --mixed_precision="no" --multi_gpu -m viewdiff.train \
accelerate launch --num_processes=1 --mixed_precision="no" -m viewdiff.train \
--finetune-config.io.pretrained_model_name_or_path $2 \
--finetune-config.io.output_dir $3 \
--finetune-config.io.experiment_name "train_teddybear" \
--finetune-config.training.mixed_precision "no" \
--finetune-config.training.dataloader_num_workers "0" \
--finetune-config.training.num_train_epochs "1000" \
--finetune-config.training.train_batch_size "1" \
--finetune-config.training.dreambooth_prior_preservation_loss_weight "0.0" \
--finetune_config.training.noise_prediction_type "epsilon" \
--finetune_config.training.prob_images_not_noisy "0.25" \
--finetune_config.training.max_num_images_not_noisy "2" \
--finetune_config.training.validation_epochs "50" \
--finetune_config.training.dreambooth_prior_preservation_every_nth "1" \
--finetune-config.optimizer.learning_rate "5e-5" \
--finetune-config.optimizer.vol_rend_learning_rate "1e-3" \
--finetune-config.optimizer.vol_rend_adam_weight_decay "0.0" \
--finetune-config.optimizer.gradient_accumulation_steps "8" \
--finetune-config.optimizer.max_grad_norm "5e-3" \
--finetune-config.cross_frame_attention.to_k_other_frames "4" \
--finetune-config.cross_frame_attention.random_others \
--finetune-config.cross_frame_attention.with_self_attention \
--finetune-config.cross_frame_attention.use_temb_cond \
--finetune-config.cross_frame_attention.mode "pretrained" \
--finetune-config.cross_frame_attention.n_cfa_down_blocks "1" \
--finetune-config.cross_frame_attention.n_cfa_up_blocks "1" \
--finetune-config.cross_frame_attention.unproj_reproj_mode "with_cfa" \
--finetune-config.cross_frame_attention.num_3d_layers "5" \
--finetune-config.cross_frame_attention.dim_3d_latent "16" \
--finetune-config.cross_frame_attention.dim_3d_grid "128" \
--finetune-config.cross_frame_attention.n_novel_images "1" \
--finetune-config.cross_frame_attention.vol_rend_proj_in_mode "multiple" \
--finetune-config.cross_frame_attention.vol_rend_proj_out_mode "multiple" \
--finetune-config.cross_frame_attention.vol_rend_aggregator_mode "ibrnet" \
--finetune-config.cross_frame_attention.last_layer_mode "no_residual_connection" \
--finetune_config.cross_frame_attention.vol_rend_model_background \
--finetune_config.cross_frame_attention.vol_rend_background_grid_percentage "0.5" \
--finetune-config.model.pose_cond_mode "sa-ca" \
--finetune-config.model.pose_cond_coord_space "absolute" \
--finetune-config.model.pose_cond_lora_rank "64" \
--finetune-config.model.n_input_images "5" \
--dataset-config.co3d-root $CO3DV2_DATASET_ROOT \
--dataset-config.category $4 \
--dataset-config.max_sequences "500" \
--dataset-config.batch.load_recentered \
--dataset-config.batch.use_blip_prompt \
--dataset-config.batch.crop "random" \
--dataset-config.batch.image_width "256" \
--dataset-config.batch.image_height "256" \
--dataset-config.batch.other_selection "mix" \
--validation-dataset-config.co3d-root $CO3DV2_DATASET_ROOT \
--validation-dataset-config.category "teddybear" \
--validation-dataset-config.max_sequences "1" \
--validation-dataset-config.batch.load_recentered \
--validation-dataset-config.batch.use_blip_prompt \
--validation-dataset-config.batch.crop "random" \
--validation-dataset-config.batch.image_width "256" \
--validation-dataset-config.batch.image_height "256" \
--validation-dataset-config.dataset_args.n_frames_per_sequence "5"
