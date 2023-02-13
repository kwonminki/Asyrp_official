#!/bin/bash

sh_file_name="script_get_lpips.sh"
gpu="0"
config="custom.yml"     # for custom.yml, you need to set custom_train_dataset_dir and custom_test_dataset_dir. If not, vice versa.
guid="smiling"          # we don't use it but need to run main.py
inv_step=1000           # if large, it takes long time.

CUDA_VISIBLE_DEVICES=$gpu python main.py --lpips                            \
                        --config $config                                    \
                        --exp ./runs/tmp                                    \
                        --edit_attr $guid                                   \
                        --do_train 1                                        \
                        --do_test 1                                         \
                        --n_train_img 100                                   \
                        --n_test_img 32                                     \
                        --t_0 999                                           \
                        --n_inv_step $inv_step                              \
                        --custom_train_dataset_dir "test_images/celeba/train"       \
                        --custom_test_dataset_dir "test_images/celeba/test"         \
                        --sh_file_name "script_get_lpips.sh"


