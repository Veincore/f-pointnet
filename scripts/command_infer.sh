#!/bin/bash
python infer/prep_data_infer.py --gpu 0 --num_point 1024 --model frustum_pointnets_v2 --model_path /media/vdc/backup/database_backup/Chris/f-pointnet/log_v2/model.ckpt --batch_size 32 --output train/detection_results_v2 --from_rgb_detection --idx_path kitti/image_sets/val.txt
