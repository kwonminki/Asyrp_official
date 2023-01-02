import argparse
import traceback
import logging
import yaml
import sys
import os
import torch
import numpy as np

from diffusion_latent import Asyrp

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    # Logging
    parser.add_argument('--sh_file_name', type=str, default='script.sh', help='copy the script this file')

    # T_edit & T_addnoise
    parser.add_argument('--user_defined_t_edit', type=int, help='if you do not use it, t_edit will be set automatically')
    parser.add_argument('--user_defined_t_addnoise', type=int, help='if you do not use it, t_addnoise will be set automatically')
    parser.add_argument('--lpips_edit_th', type=float, default=0.33, help='we use lpips_edit_th to get t_edit')
    parser.add_argument('--lpips_addnoise_th', type=float, default=0.1, help='we use lpips_addnoise_th to get t_addnoise')

    parser.add_argument('--add_noise_from_xt', action='store_true', help='add_noise_from_xt')
    parser.add_argument('--origin_process_addnoise', action='store_true', help='origin_process_addnoise')


    # Training Mode
    parser.add_argument('--run_train', action='store_true', help='run_train')
    parser.add_argument('--train_delta_block', action='store_true', help='train Delta_block')
    parser.add_argument('--train_delta_h', action='store_true', help='train Delta_h')
    parser.add_argument('--image_space_noise_optim', action='store_true', help='train image_space_noise_optim')
    parser.add_argument('--image_space_noise_optim_delta_block', action='store_true', help='image_space_noise_optim_delta_block')
    parser.add_argument('--just_precompute', action='store_true', help='just_precompute')
    parser.add_argument('--ignore_timesteps', action='store_true', default=False, help='train without timesteps')
    parser.add_argument('--use_id_loss', action='store_true', default=False, help='train with id loss')
    parser.add_argument('--shuffle_train_dataloader', action='store_true', default=False, help='shuffle train dataloader')
    parser.add_argument('--re_precompute', action='store_true', default=False, help='re-precompute')
    parser.add_argument('--save_checkpoint_only_last_iter', action='store_true', default=False, help='carefully')
    parser.add_argument('--save_checkpoint_during_iter', action='store_true', default=False, help='carefully')
    parser.add_argument('--save_checkpoint_step', type=int, default=200, help='save checkpoint every save_checkpoint_step')
    parser.add_argument('--start_iter_when_you_use_pretrained', type=int, default=0, help='start_iter_when_you_use_pretrained')
    parser.add_argument('--image_space_noise_optim_origin', action='store_true')

    # Training details
    parser.add_argument('--lr_training', type=float, default=2e-1, help='Initial learning rate for training')
    parser.add_argument('--use_x0_tensor', action='store_true', help='use_x0_tensor')
    parser.add_argument('--save_x0', action='store_true', help='save x0_tensor (original image)')
    parser.add_argument('--save_x_origin', action='store_true', help='save x_origin (original DDIM processing)')
    
    parser.add_argument('--custom_train_dataset_dir', type=str, default="./custom/train")
    parser.add_argument('--custom_test_dataset_dir', type=str, default="./custom/test")

    # Test Mode
    parser.add_argument('--run_test', action='store_true', help='run_test')
    parser.add_argument('--load_random_noise', action='store_true', help='run_test')
    parser.add_argument('--saved_random_noise', action='store_true', help='run_test')
    
    parser.add_argument('--delta_interpolation', action='store_true', help='run_test')
    parser.add_argument('--max_delta', type=float, default=1.0, help='max delta for evaluating the generative process')
    parser.add_argument('--min_delta', type=float, default=0.0, help='min delta for evaluating the generative process')
    parser.add_argument('--num_delta', type=int, default=5, help='num of delta for evaluating the generative process')
    
    parser.add_argument('--hs_coeff_delta_h', type=float, default=1.0, help='max delta for evaluating the generative process')
    parser.add_argument('--hs_coeff_origin_h', type=float, default=1.0, help='max delta for evaluating the generative process')
    
    parser.add_argument('--target_image_id', type=str, help='Sampling only one image which is target_image_id')
    parser.add_argument('--start_image_id', type=int, default=0, help='Sampling after start_image_id')
    
    parser.add_argument('--save_process_origin', action='store_true', help='save_origin_process')
    parser.add_argument('--save_process_delta_h', action='store_true', help='save_delta_h_process')
    
    parser.add_argument('--num_mean_of_delta_hs', type=int, default=0, help='Get mean of delta_h from num of data')

    parser.add_argument('--multiple_attr', type=str, default='', help='multiple attr for evaluating the generative process')
    parser.add_argument('--multiple_hs_coeff', type=str, default='', help='multiple coeffs')
    parser.add_argument('--masked_h', type=str, default='', help='')

    parser.add_argument('--manual_checkpoint_name', type=str, default="", help='manually choose the name of chekcpoint')
    parser.add_argument('--choose_checkpoint_num', type=str, default='', help='if model is saved during an iteration, you can choose the number of chekcpoint of the model. This is for training from random noise')
    parser.add_argument('--load_from_checkpoint', type=str)

    parser.add_argument('--do_alternate', type=int, default=0, help='Whether to train or not during CLIP finetuning')
    parser.add_argument('--pass_editing', action='store_true', help='Whether to train or not during CLIP finetuning')
    
    # Style Transfer Mode
    parser.add_argument('--style_transfer', action="store_true")
    parser.add_argument('--style_transfer_style_from_train_images', default=False, action="store_true")
    parser.add_argument('--style_transfer_noise_from', type=str, default="contents")


    # LPIPS
    parser.add_argument('--lpips', action="store_true")
    parser.add_argument('--custom_dataset_name', type=str, default="celeba")

    # Additional test
    parser.add_argument('--latent_classifier', action="store_true")
    parser.add_argument('--warigari', type=float, default=0.0)
    parser.add_argument('--attr_index', type=int)
    parser.add_argument('--classification_results_file_name', type=str, default="classification_results")
    parser.add_argument('--DirectionalClipSmilarity', action="store_true")

    # Mode
    parser.add_argument('--clip_finetune', action='store_true')
    parser.add_argument('--global_clip', action='store_true')
    parser.add_argument('--run_origin', action='store_true')
    parser.add_argument('--latent_at', action='store_true')
    parser.add_argument('--test_celeba_dialog', action='store_true')
    
    parser.add_argument('--latent_clr', action='store_true')
    parser.add_argument('--eval_latent_clr', action='store_true')
    parser.add_argument('--interpolation', action='store_true')
    parser.add_argument('--interpolation2', action='store_true')
    parser.add_argument('--clip_latent_optim', action='store_true')
    parser.add_argument('--edit_images_from_dataset', action='store_true')
    parser.add_argument('--edit_one_image', action='store_true')
    parser.add_argument('--unseen2unseen', action='store_true')
    parser.add_argument('--clip_finetune_eff', action='store_true')
    parser.add_argument('--edit_one_image_eff', action='store_true')
    parser.add_argument('--save_precomputed_images', action='store_true')
    parser.add_argument('--test_pretrained', action='store_true')
    parser.add_argument('--compute_distance_graph', action="store_true")

    parser.add_argument('--global_cliploss', action="store_true")

    parser.add_argument('--save_to_folder', type=str)
    parser.add_argument('--from_noise', action="store_true")
    
    parser.add_argument('--random_ddim', action="store_true")
    parser.add_argument('--direct_ddim', action="store_true")
    parser.add_argument('--direct_same_regardless_of_t', action="store_true")
    parser.add_argument('--step_40_to_ddpm', action="store_true")
    parser.add_argument('--l1_loss_with_x0',default=False, action="store_true", help="if false, l1 loss is with origin_predicted_x0")

    parser.add_argument('--pass_origin_and_save_real_image',default=False, action="store_true")
    parser.add_argument('--style_transfer_use_mean',default=False, action="store_true", help="if false, use adain (default)")
    parser.add_argument('--analysis',default=False, action="store_true", help="dont save")
    parser.add_argument('--Three_compare_addnoise', action="store_true")
    parser.add_argument('--run_optimize_delta_h', action="store_true")
    parser.add_argument('--run_test_pretrained_self_delta_h', action="store_true")
    parser.add_argument('--test_pretrained_at_once', action="store_true")

    # Default
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--exp', type=str, default='./runs/', help='Path for saving running related data.')
    parser.add_argument('--comment', type=str, default='', help='A string for experiment comment')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('--ni', type=int, default=1,  help="No interaction. Suitable for Slurm Job launcher")
    parser.add_argument('--align_face', type=int, default=1, help='align face or not')

    # Text
    parser.add_argument('--edit_attr', type=str, default=None, help='Attribute to edit defiend in ./utils/text_dic.py')
    parser.add_argument('--src_txts', type=str, action='append', help='Source text e.g. Face')
    parser.add_argument('--trg_txts', type=str, action='append', help='Target text e.g. Angry Face')
    parser.add_argument('--target_class_num', type=str, default=None)

    # Sampling
    parser.add_argument('--t_0', type=int, default=999, help='Return step in [0, 1000)')
    parser.add_argument('--n_inv_step', type=int, default=40, help='# of steps during generative pross for inversion')
    parser.add_argument('--n_train_step', type=int, default=6, help='# of steps during generative pross for train')
    parser.add_argument('--n_test_step', type=int, default=40, help='# of steps during generative pross for test')
    parser.add_argument('--sample_type', type=str, default='ddim', help='ddpm for Markovian sampling, ddim for non-Markovian sampling')
    parser.add_argument('--eta', type=float, default=0.0, help='Controls of varaince of the generative process')
    parser.add_argument('--rambda', type=float, default=1.0, help='Controls of rambda')

    parser.add_argument('--LPIPS_addnoise_th', type=float, default=0.1, help='LPIPS_addnoise_th')
    parser.add_argument('--n_test_pretrained_inv_step', type=int, default=40, help='# of steps during generative pross for inversion')


    # Train & Test
    parser.add_argument('--do_train', type=int, default=1, help='Whether to train or not during CLIP finetuning')
    parser.add_argument('--retrain', type=int, default=0, help='Whether to train or not during CLIP finetuning')
    parser.add_argument('--do_test', type=int, default=1, help='Whether to test or not during CLIP finetuning')
    parser.add_argument('--save_train_image', type=int, default=1, help='Wheter to save training results during CLIP fineuning')
    parser.add_argument('--save_train_image_step', type=int, default=4, help='Wheter to save training results during CLIP fineuning')
    parser.add_argument('--save_train_image_iter', type=int, default=1, help='Wheter to save training results during CLIP fineuning')
    parser.add_argument('--bs_train', type=int, default=1, help='Training batch size during CLIP fineuning')
    parser.add_argument('--bs_test', type=int, default=1, help='Test batch size during CLIP fineuning')
    parser.add_argument('--n_precomp_img', type=int, default=100, help='# of images to precompute latents')
    parser.add_argument('--n_train_img', type=int, default=50, help='# of training images')
    parser.add_argument('--n_test_img', type=int, default=10, help='# of test images')
    parser.add_argument('--model_path', type=str, default=None, help='Test model path')
    parser.add_argument('--img_path', type=str, default=None, help='Image path to test')
    parser.add_argument('--deterministic_inv', type=int, default=1, help='Whether to use deterministic inversion during inference')
    parser.add_argument('--hybrid_noise', type=int, default=0, help='Whether to change multiple attributes by mixing multiple models')
    parser.add_argument('--get_h_num', type=int, default=0, help='Training batch size during Latent CLR')
    parser.add_argument('--model_ratio', type=float, default=1, help='Degree of change, noise ratio from original and finetuned model.')
    

    # DiffStyle
    parser.add_argument('--diff_style', action="store_true")
    parser.add_argument('--content_dir', type=str, default="./source_images/content", help='Path to the content images')
    parser.add_argument('--style_dir', type=str, default="./source_images/style", help='Path to the style images')
    parser.add_argument('--save_dir', type=str, default="./results")
    parser.add_argument('--n_gen_step' , type=int, default=1000, help='Number of steps for generating images')
    parser.add_argument('--content_replace_step', type=int, default=50, help='Number of steps for replacing content images')
    parser.add_argument('--hs_coeff', type=float, default=0.9, help='hs coefficient')
    parser.add_argument('--use_mask', action="store_true", help='use mask or not')
    parser.add_argument('--dt_lambda', type=float, default=1.0, help='dt lambda coefficient for sampling calibration')
    parser.add_argument('--dt_end', type=int, default = 950, help='dt end')
    parser.add_argument('--t_noise', type=int, default=0, help='quality boosting')

    
    # parser.add_argument('--save_grid', action="store_true", help="save all results in a grid image")
    
    # Loss & Optimization
    parser.add_argument('--clip_loss_w', type=float, default=3, help='Weights of CLIP loss')
    parser.add_argument('--clr_loss_w', type=int, default=3, help='Weights of CLIP loss')
    parser.add_argument('--l1_loss_w', type=float, default=0, help='Weights of L1 loss')
    parser.add_argument('--id_loss_w', type=float, default=0, help='Weights of ID loss')
    parser.add_argument('--clip_model_name', type=str, default='ViT-B/16', help='ViT-B/16, ViT-B/32, RN50x16 etc')
    parser.add_argument('--lr_clip_finetune', type=float, default=2e-6, help='Initial learning rate for finetuning')
    parser.add_argument('--lr_latent_clr', type=float, default=2e-6, help='Initial learning rate for latent clr')
    parser.add_argument('--lr_clip_lat_opt', type=float, default=2e-2, help='Initial learning rate for latent optim')
    parser.add_argument('--n_iter', type=int, default=1, help='# of iterations of a generative process with `n_train_img` images')
    parser.add_argument('--scheduler', type=int, default=1, help='Whether to increase the learning rate')
    parser.add_argument('--scheduler_step_size', type=int, default=3, help='Whether to increase the learning rate')
    parser.add_argument('--sch_gamma', type=float, default=0.1, help='Scheduler gamma')

    parser.add_argument('--var', type=int, default=100, help='Using for debug')
    parser.add_argument('--maintain', type=int, default=400, help='Using for debug')
    parser.add_argument('--interpolation_step', type=int, default=4, help='Using for debug')
    parser.add_argument('--maintain_min',type=int,default=50,help = '')

    parser.add_argument('--get_SNR', action="store_true", default=False, help='Whether to get SNR')

    args = parser.parse_args()

    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    args.exp = args.exp + f'_LC_{new_config.data.category}_t{args.t_0}_ninv{args.n_inv_step}_ngen{args.n_train_step}'


    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    os.makedirs('checkpoint', exist_ok=True)
    os.makedirs('checkpoint_latent', exist_ok=True)
    os.makedirs('precomputed', exist_ok=True)
    os.makedirs('runs', exist_ok=True)
    os.makedirs(args.exp, exist_ok=True)

    import shutil
    if args.run_test:
        shutil.copy(args.sh_file_name, os.path.join(args.exp, f"{(args.sh_file_name).split('.')[0]}_test.sh"))
    elif args.style_transfer:
        shutil.copy(args.sh_file_name, os.path.join(args.exp, f"{(args.sh_file_name).split('.')[0]}_style_transfer.sh"))
    elif args.run_train:
        shutil.copy(args.sh_file_name, os.path.join(args.exp, f"{args.sh_file_name.split('.')[0]}_train.sh"))
    elif args.lpips:
        pass

    args.training_image_folder = os.path.join(args.exp, 'training_images')
    if not os.path.exists(args.training_image_folder):
        os.makedirs(args.training_image_folder)
    
    args.test_image_folder = os.path.join(args.exp, 'test_images', str(args.n_test_step))
    if not os.path.exists(args.test_image_folder):
        os.makedirs(args.test_image_folder)  

    args.image_folder = os.path.join(args.exp, 'image_samples')
    if not os.path.exists(args.image_folder):
        os.makedirs(args.image_folder)
    else:
        overwrite = False
        if args.ni:
            overwrite = True
        else:
            response = input("Image folder already exists. Overwrite? (Y/N)")
            if response.upper() == 'Y':
                overwrite = True

        if overwrite:
            # shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder, exist_ok=True)
        else:
            print("Output image folder exists. Program halted.")
            sys.exit(0)

    if args.save_to_folder:
        args.training_image_folder = args.save_to_folder

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()

    # This code is for me. If you don't need it, just remove it out.
    if torch.cuda.is_available():
        assert args.bs_train % torch.cuda.device_count() == 0, f"Number of GPUs ({torch.cuda.device_count()}) must be a multiple of batch size ({args.bs_train})"

    runner = Asyrp(args, config) # if you want to specify the device, add device="something" in the argument
    try:
        # check the example script files for essential parameters
        if args.run_train:
            runner.run_training()
        elif args.run_test:
            runner.run_test()
        elif args.lpips:
            runner.compute_lpips_distance()

    except Exception:
        logging.error(traceback.format_exc())

    return 0


if __name__ == '__main__':
    sys.exit(main())
