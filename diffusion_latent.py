from audioop import reverse
from genericpath import isfile
import time
from glob import glob
from models.guided_diffusion.script_util import guided_Diffusion
from models.improved_ddpm.nn import normalization
from tqdm import tqdm
import os
import numpy as np
import cv2
from PIL import Image
import torch
from torch import nn
import torchvision.utils as tvu
from torchvision import models
import torchvision.transforms as transforms
import torch.nn.functional as F
from losses.clip_loss import CLIPLoss
import random
import copy

from models.ddpm.diffusion import DDPM
from models.improved_ddpm.script_util import i_DDPM
from utils.diffusion_utils import get_beta_schedule, denoising_step
from utils.text_dic import SRC_TRG_TXT_DIC
from losses import id_loss
from datasets.data_utils import get_dataset, get_dataloader
from configs.paths_config import DATASET_PATHS, MODEL_PATHS
from datasets.imagenet_dic import IMAGENET_DIC

class Asyrp(object):
    def __init__(self, args, config, device=None):
        # ----------- predefined parameters -----------#
        self.args = args
        self.config = config
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = betas * \
                             (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.alphas_cumprod = alphas_cumprod

        if self.model_var_type == "fixedlarge":
            self.logvar = np.log(np.append(posterior_variance[1], betas[1:]))

        elif self.model_var_type == 'fixedsmall':
            self.logvar = np.log(np.maximum(posterior_variance, 1e-20))

        self.learn_sigma = False # it will be changed in load_pretrained_model()

        # ----------- Editing txt -----------#
        if self.args.edit_attr is None:
            self.src_txts = self.args.src_txts
            self.trg_txts = self.args.trg_txts
        elif self.args.edit_attr == "attribute":
            pass
        else:
            self.src_txts = SRC_TRG_TXT_DIC[self.args.edit_attr][0]
            self.trg_txts = SRC_TRG_TXT_DIC[self.args.edit_attr][1]


    def load_pretrained_model(self):

        # ----------- Model -----------#
        if self.config.data.dataset == "LSUN":
            if self.config.data.category == "bedroom":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/bedroom.ckpt"
            elif self.config.data.category == "church_outdoor":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/church_outdoor.ckpt"
        elif self.config.data.dataset in ["CelebA_HQ", "CUSTOM", "CelebA_HQ_Dialog"]:
            url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt"
        elif self.config.data.dataset in ["FFHQ", "AFHQ", "IMAGENET", "MetFACE"]:
            # get the model ["FFHQ", "AFHQ", "MetFACE"] from 
            # https://1drv.ms/u/s!AkQjJhxDm0Fyhqp_4gkYjwVRBe8V_w?e=Et3ITH
            # reference : ILVR (https://arxiv.org/abs/2108.02938), P2 weighting (https://arxiv.org/abs/2204.00227)
            # reference github : https://github.com/jychoi118/ilvr_adm , https://github.com/jychoi118/P2-weighting 

            # get the model "IMAGENET" from
            # https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt
            # reference : ADM (https://arxiv.org/abs/2105.05233)
            pass
        else:
            # if you want to use LSUN-horse, LSUN-cat -> https://github.com/openai/guided-diffusion
            # if you want to use CUB, Flowers -> https://1drv.ms/u/s!AkQjJhxDm0Fyhqp_4gkYjwVRBe8V_w?e=Et3ITH
            raise ValueError

        if self.config.data.dataset in ["CelebA_HQ", "LSUN", "CelebA_HQ_Dialog"]:
            model = DDPM(self.config) 
            if self.args.model_path:
                init_ckpt = torch.load(self.args.model_path)
            else:
                init_ckpt = torch.hub.load_state_dict_from_url(url, map_location=self.device)
            self.learn_sigma = False
            print("Original diffusion Model loaded.")
        elif self.config.data.dataset in ["FFHQ", "AFHQ", "IMAGENET"]:
            model = i_DDPM(self.config.data.dataset) #Get_h(self.config, model="i_DDPM", layer_num=self.args.get_h_num) #
            if self.args.model_path:
                init_ckpt = torch.load(self.args.model_path)
            else:
                init_ckpt = torch.load(MODEL_PATHS[self.config.data.dataset])
            self.learn_sigma = True
            print("Improved diffusion Model loaded.")
        elif self.config.data.dataset in ["MetFACE", "CelebA_HQ_P2"]:
            model = guided_Diffusion(self.config.data.dataset)
            init_ckpt = torch.load(MODEL_PATHS[self.config.data.dataset])
            self.learn_sigma = True
        else:
            print('Not implemented dataset')
            raise ValueError
        model.load_state_dict(init_ckpt, strict=False)

        return model

    
    def run_training(self):
        print("Running Training...")

        # ----------- Losses -----------#
        # We tried to use ID loss and it works well.
        # But it is not used in the paper because it is not necessary.
        # We just leave the code here for future research.
        if self.args.use_id_loss:
            id_loss_func = id_loss.IDLoss().to(self.device)
        
        # Set self.t_edit & self.t_addnoise & return cosine similarity of attribute
        cosine, clip_loss_func = self.set_t_edit_t_addnoise(LPIPS_th=self.args.lpips_edit_th, 
                                                            LPIPS_addnoise_th=self.args.lpips_addnoise_th,
                                                            return_clip_loss=True)
        clip_loss_func = clip_loss_func.to(self.device)
        
        # For memory
        for p in clip_loss_func.parameters():
            p.requires_grad = False
        for p in clip_loss_func.model.parameters():
            p.requires_grad = False

        # ----------- Get seq -----------#    
        if self.args.n_train_step != 0:
            # do not need to train T~0
            seq_train = np.linspace(0, 1, self.args.n_train_step) * self.args.t_0
            seq_train = seq_train[seq_train >= self.t_edit]
            seq_train = [int(s+1e-6) for s in list(seq_train)] # for float to int
            print('Uniform skip type')
        else:
            seq_train = list(range(self.t_edit, self.args.t_0))
            print('No skip')
        seq_train_next = [-1] + list(seq_train[:-1])

        # it is for sampling
        seq_test = np.linspace(0, 1, self.args.n_train_step) * self.args.t_0
        seq_test = [int(s+1e-6) for s in list(seq_test)]
        seq_test_next = [-1] + list(seq_test[:-1])

        # ----------- Model -----------#
        model = self.load_pretrained_model()
        optim_param_list = []
        delta_h_dict = {}
        for i in seq_train:
            delta_h_dict[i] = None

        if self.args.train_delta_block:
            model.setattr_layers(self.args.get_h_num)
            print("Setattr layers")
            model = model.to(self.device)
            model = torch.nn.DataParallel(model)

            for i in range(self.args.get_h_num):
                get_h = getattr(model.module, f"layer_{i}")
                optim_param_list = optim_param_list + list(get_h.parameters())

        elif self.args.train_delta_h:
            # h_dim is hard coded to be 512
            # It can be converted to get automatically
            if self.args.ignore_timesteps:
                delta_h_dict[0] = torch.nn.Parameter(torch.randn((512, 8, 8))*0.2) # initialization of delta_h
            else:
                for i in seq_train:
                    delta_h_dict[i] = torch.nn.Parameter(torch.randn((512, 8, 8))*0.2) # initialization of delta_h

            model = model.to(self.device)
            model = torch.nn.DataParallel(model)

            for key in delta_h_dict.keys():
                optim_param_list = optim_param_list + [delta_h_dict[key]]
            
        # optim_ft = torch.optim.Adam(optim_get_h_list, weight_decay=0, lr=self.args.lr_latent_clr)
        optim_ft = torch.optim.SGD(optim_param_list, weight_decay=0, lr=self.args.lr_training)
        scheduler_ft = torch.optim.lr_scheduler.StepLR(optim_ft, step_size=self.args.scheduler_step_size, gamma=self.args.sch_gamma)
        print(f"Setting optimizer with lr={self.args.lr_training}")

        # hs_coeff[0] is for original h, hs_coeff[1] is for delta_h
        # if you want to train multiple delta_h at once, you have to modify this part.
        hs_coeff = (1.0, 1.0)

        # ----------- Pre-compute -----------#
        print("Prepare identity latent...")
        if self.args.load_random_noise:
            # get Random noise xT
            img_lat_pairs_dic = self.random_noise_pairs(model, saved_noise=self.args.saved_random_noise, save_imgs=self.args.save_precomputed_images)
        else:
            # get Real image xT
            img_lat_pairs_dic = self.precompute_pairs(model, self.args.save_precomputed_images)
        
        if self.args.just_precompute:
            # if you just want to precompute, you can stop here.
            print("Pre-computed done.")
            return
        
        # if you want to train with specific image, you can use this part.
        if self.args.target_image_id:
            self.args.target_image_id = self.args.target_image_id.split(" ")
            self.args.target_image_id = [int(i) for i in self.args.target_image_id]

        # ----------- Training -----------#
        for it_out in range(self.args.start_iter_when_you_use_pretrained ,self.args.n_iter):
            exp_id = os.path.split(self.args.exp)[-1]
            if self.args.load_from_checkpoint:
                save_name = f'checkpoint/{self.args.load_from_checkpoint}_LC_{self.config.data.category}_t{self.args.t_0}_ninv{self.args.n_inv_step}_ngen{self.args.n_train_step}_{it_out}.pth'
            else:
                save_name = f'checkpoint/{exp_id}_{it_out}.pth'

            # train set
            if self.args.do_train:
                save_image_iter = 0
                save_model_iter_from_noise = 0
                if self.args.retrain==0 and os.path.exists(save_name):
                    # load checkpoint
                    print(f'{save_name} already exists. load checkpoint')
                    self.args.retrain = 0
                    optim_ft.load_state_dict(torch.load(save_name)["optimizer"])
                    scheduler_ft.load_state_dict(torch.load(save_name)["scheduler"])
                    scheduler_ft.step()
                    #print lr of now
                    print(f"Loaded lr={optim_ft.param_groups[0]['lr']}")
                    # get_h_num default is 0;
                    if self.args.train_delta_block:
                        for i in range(self.args.get_h_num):
                            get_h = getattr(model.module, f"layer_{i}")
                            get_h.load_state_dict(torch.load(save_name)[f"{i}"])
                    if self.args.train_delta_h:
                        for i in delta_h_dict.keys():
                            delta_h_dict[i] = torch.load(save_name)[f"{i}"]
                    continue
                else:
                    # Unfortunately, ima_lat_pairs_dic does not match with batch_size
                    # I'm sorry but you have to get ima_lat_pairs_dic with batch_size == 1
                    x_lat_tensor = None
                    x0_tensor = None

                    for step, (x0, _, x_lat) in enumerate(img_lat_pairs_dic['train']):
                        if self.args.target_image_id:
                            assert self.args.bs_train == 1, "target_image_id is only supported for batch_size == 1"
                            if not step in self.args.target_image_id:
                                continue
                        if x_lat_tensor is None:
                            x_lat_tensor = x_lat
                            if self.args.use_x0_tensor:
                                x0_tensor = x0
                        else:
                            x_lat_tensor = torch.cat((x_lat_tensor, x_lat), dim=0)
                            if self.args.use_x0_tensor:
                                x0_tensor = torch.cat((x0_tensor, x0), dim=0)
                        if (step+1) % self.args.bs_train != 0:
                            continue
                        # LoL. now x_lat_tensor has batch_size == bs_train

                        # torch.cuda.empty.cache()
                        model.train()
                        # For memory
                        for p in model.module.parameters():
                            p.requires_grad = False
                        if self.args.train_delta_block:
                            for i in range(self.args.get_h_num):
                                get_h = getattr(model.module, f"layer_{i}")
                                for p in get_h.parameters():
                                    p.requires_grad = True

                        time_in_start = time.time()

                        # original DDIM
                        x_origin = x_lat_tensor.to(self.device)
                        # editing by Asyrp
                        xt_next = x_lat_tensor.to(self.device)
                        
                        # Finally, go into training
                        with tqdm(total=len(seq_train), desc=f"training iteration") as progress_bar:
                            for t_it, (i, j) in enumerate(zip(reversed(seq_train), reversed(seq_train_next))):
    
                                optim_ft.zero_grad()
                                t = (torch.ones(self.args.bs_train) * i).to(self.device)
                                t_next = (torch.ones(self.args.bs_train) * j).to(self.device)
                                
                                # step 1: Asyrp
                                xt_next, x0_t, _, _ = denoising_step(xt_next.detach(), t=t, t_next=t_next, models=model,
                                                            logvars=self.logvar,                                        
                                                            b=self.betas,
                                                            sampling_type=self.args.sample_type,
                                                            eta=0.0,
                                                            learn_sigma=self.learn_sigma,
                                                            index=0 if not (self.args.image_space_noise_optim or self.args.image_space_noise_optim_delta_block) else None,
                                                            t_edit = self.t_edit,
                                                            hs_coeff=hs_coeff,
                                                            delta_h= delta_h_dict[0] if (self.args.ignore_timesteps and self.args.train_delta_h) else delta_h_dict[t[0].item()],
                                                            ignore_timestep=self.args.ignore_timesteps,
                                                            )
                                                            # when train delta_block, delta_h is None (ignored)
                                # step 2: DDIM
                                with torch.no_grad():    
                                    x_origin, x0_t_origin, _, _ = denoising_step(x_origin.detach(), t=t, t_next=t_next, models=model,
                                                                    logvars=self.logvar,
                                                                    b=self.betas,
                                                                    sampling_type=self.args.sample_type,                                                                
                                                                    eta=0.0,
                                                                    learn_sigma=self.learn_sigma,
                                                                    )

                                progress_bar.update(1)
                                
                                loss = 0
                                loss_id = 0
                                loss_l1 = 0
                                loss_clr = 0
                                loss_clip = 0

                                # L1 loss
                                loss_l1 += nn.L1Loss()(x0_t, x0_t_origin)

                                # Following DiffusionCLIP, we use direction clip loss as below
                                loss_clip = -torch.log((2 - clip_loss_func(x0, self.src_txts[0], x0_t, self.trg_txts[0])) / 2)
                                
                                if self.args.use_id_loss:
                                    # We don't use this.
                                    loss_id += torch.mean(id_loss_func(x0_t, x0_t_origin))

                                loss += self.args.id_loss_w * loss_id
                                loss += self.args.l1_loss_w * loss_l1 * cosine
                                loss += self.args.clip_loss_w * loss_clip

                                loss.backward()
                                optim_ft.step()   

                                progress_bar.set_description(f"{step}-{it_out}: loss_clr: {loss_clr:.3f} loss_l1: {loss_l1:.3f} loss_id: {loss_id:.3f} loss_clip:{loss_clip} loss: {loss:.3f} ")

                        # save image
                        if self.args.save_train_image and save_image_iter % self.args.save_train_image_step == 0 and it_out % self.args.save_train_image_iter == 0:
                            self.save_image(model, x_lat_tensor, seq_test, seq_test_next,
                                            save_x0 = self.args.save_x0, save_x_origin = self.args.save_x_origin,
                                            x0_tensor=x0_tensor, delta_h_dict=delta_h_dict,
                                            folder_dir=self.args.training_image_folder,
                                            file_name=f'train_{step}_{it_out}', hs_coeff=hs_coeff,
                                            )
                        
                        if self.args.save_checkpoint_during_iter and save_image_iter % self.args.save_checkpoint_step ==0:
                            dicts = {}
                            if self.args.train_delta_block:
                                for i in range(self.args.get_h_num):
                                    get_h = getattr(model.module, f"layer_{i}")
                                    dicts[f"{i}"] = get_h.state_dict()
                            if self.args.train_delta_h:
                                for key in delta_h_dict.keys():
                                    dicts[f"{key}"] = delta_h_dict[key]

                            save_name_tmp = save_name.split('.pth')[0] + "_" + str(save_model_iter_from_noise) + '.pth'
                            torch.save(dicts, save_name_tmp)
                            print(f'Model {save_name_tmp} is saved.')

                            save_model_iter_from_noise += 1
                                                                
                        time_in_end = time.time()
                        print(f"Training for 1 step {time_in_end - time_in_start:.4f}s")
                        if step == self.args.n_train_img - 1:
                            break
                        save_image_iter += 1
                        x_lat_tensor = None
                        x0_tensor = None


                    # ------------------ Save ------------------#
                    dicts = {}
                    if self.args.train_delta_block:
                        for i in range(self.args.get_h_num):
                            get_h = getattr(model.module, f"layer_{i}")
                            dicts[f"{i}"] = get_h.state_dict()
                    if self.args.train_delta_h:
                        for key in delta_h_dict.keys():
                            dicts[f"{key}"] = delta_h_dict[key]
                    
                    dicts["optimizer"] = optim_ft.state_dict()
                    dicts["scheduler"] = scheduler_ft.state_dict()
                    torch.save(dicts, save_name)
                    print(f'Model {save_name} is saved.')
                    scheduler_ft.step()

                    if self.args.save_checkpoint_only_last_iter:
                        if os.path.exists(f'checkpoint/{exp_id}_{it_out - 1}.pth'):
                            os.remove(f'checkpoint/{exp_id}_{it_out - 1}.pth')

        # ------------------ Test ------------------#
        if self.args.do_test:
            x_lat_tensor = None
            x0_tensor = None

            for step, (x0, _, x_lat) in enumerate(img_lat_pairs_dic['test']):
                    
                if x_lat_tensor is None:
                    x_lat_tensor = x_lat
                    if self.args.use_x0_tensor:
                        x0_tensor = x0
                else:
                    x_lat_tensor = torch.cat((x_lat_tensor, x_lat), dim=0)
                    if self.args.use_x0_tensor:
                        x0_tensor = torch.cat((x0_tensor, x0), dim=0)
                if (step+1) % self.args.bs_train != 0:
                    continue

                self.save_image(model, x_lat_tensor, seq_test, seq_test_next,
                                            save_x0 = self.args.save_x0, save_x_origin = self.args.save_x_origin,
                                            x0_tensor=x0_tensor, delta_h_dict=delta_h_dict,
                                            folder_dir=self.args.test_image_folder,
                                            file_name=f'test_{step}_{self.args.n_iter - 1}', hs_coeff=hs_coeff,
                                            )
                                        
                if step == self.args.n_test_img - 1:
                    break
                save_image_iter += 1
                x_lat_tensor = None
                x0_tensor = None



    @torch.no_grad()
    def save_image(self, model, x_lat_tensor, seq_inv, seq_inv_next,
                    save_x0 = False, save_x_origin = False,
                    save_process_delta_h = False, save_process_origin = False,
                    x0_tensor = None, delta_h_dict=None, get_delta_hs=False,
                    folder_dir="", file_name="", hs_coeff=(1.0,1.0),
                    image_space_noise_dict=None):
        
        if save_process_origin or save_process_delta_h:
            os.makedirs(os.path.join(folder_dir,file_name), exist_ok=True)

        process_num = int(save_x_origin) + (len(hs_coeff) if isinstance(hs_coeff, list) else 1)
        

        with tqdm(total=len(seq_inv)*(process_num), desc=f"Generative process") as progress_bar:
            time_s = time.time()

            x_list = []

            if save_x0:
                if x0_tensor is not None:
                    x_list.append(x0_tensor.to(self.device))
            
            if save_x_origin:
            # No delta h
                x = x_lat_tensor.clone().to(self.device)

                for it, (i, j) in enumerate(zip(reversed((seq_inv)), reversed((seq_inv_next)))):
                    t = (torch.ones(self.args.bs_train) * i).to(self.device)
                    t_next = (torch.ones(self.args.bs_train) * j).to(self.device)

                    x, x0_t, _, _  = denoising_step(x, t=t, t_next=t_next, models=model,
                                    logvars=self.logvar,
                                    sampling_type= self.args.sample_type,
                                    b=self.betas,
                                    learn_sigma=self.learn_sigma,
                                    eta=1.0 if (self.args.origin_process_addnoise and t[0]<self.t_addnoise) else 0.0,
                                    )
                    progress_bar.update(1)
                    
                    if save_process_origin:
                        output = torch.cat([x, x0_t], dim=0)
                        output = (output + 1) * 0.5
                        grid = tvu.make_grid(output, nrow=self.args.bs_train, padding=1)
                        tvu.save_image(grid, os.path.join(folder_dir, file_name, f'origin_{int(t[0].item())}.png'), normalization=True)

                x_list.append(x)

            if self.args.pass_editing:
                pass
            else:
                if not isinstance(hs_coeff, list):
                    hs_coeff = [hs_coeff]
                
                for hs_coeff_tuple in hs_coeff:

                    x = x_lat_tensor.clone().to(self.device)

                    for it, (i, j) in enumerate(zip(reversed((seq_inv)), reversed((seq_inv_next)))):
                        t = (torch.ones(self.args.bs_train) * i).to(self.device)
                        t_next = (torch.ones(self.args.bs_train) * j).to(self.device)

                        x, x0_t, delta_h, _ = denoising_step(x, t=t, t_next=t_next, models=model,
                                        logvars=self.logvar,                                
                                        sampling_type=self.args.sample_type,
                                        b=self.betas,
                                        learn_sigma=self.learn_sigma,
                                        index=self.args.get_h_num-1 if not (self.args.image_space_noise_optim or self.args.image_space_noise_optim_delta_block) else None,
                                        eta=1.0 if t[0]<self.t_addnoise else 0.0,
                                        t_edit= self.t_edit,
                                        hs_coeff=hs_coeff_tuple,
                                        delta_h=None if get_delta_hs else delta_h_dict[0] if (self.args.ignore_timesteps and self.args.train_delta_h) else delta_h_dict[int(t[0].item())] if t[0]>= self.t_edit else None,
                                        ignore_timestep=self.args.ignore_timesteps,
                                        dt_lambda=self.args.dt_lambda,
                                        warigari=self.args.warigari,
                                        )
                        progress_bar.update(1)

                        if save_process_delta_h:
                            output = torch.cat([x, x0_t], dim=0)
                            output = (output + 1) * 0.5
                            grid = tvu.make_grid(output, nrow=self.args.bs_train, padding=1)
                            tvu.save_image(grid, os.path.join(folder_dir, file_name, f'delta_h_{int(t[0].item())}.png'), normalization=True)
                        if get_delta_hs and t[0]>= self.t_edit:
                            if delta_h_dict[t[0].item()] is None:
                                delta_h_dict[t[0].item()] = delta_h
                            else:
                                delta_h_dict[int(t[0].item())] = delta_h_dict[int(t[0].item())] + delta_h

                    x_list.append(x)

        x = torch.cat(x_list, dim=0)
        x = (x + 1) * 0.5

        grid = tvu.make_grid(x, nrow=self.args.bs_train, padding=1)

        tvu.save_image(grid, os.path.join(folder_dir, f'{file_name}_ngen{self.args.n_train_step}.png'), normalization=True)

        time_e = time.time()
        print(f'{time_e - time_s} seconds, {file_name}_ngen{self.args.n_train_step}.png is saved')

    # test
    @torch.no_grad()
    def run_test(self):
        print("Running Test")

        # Set self.t_edit & self.t_addnoise & return cosine similarity of attribute
        cosine = self.set_t_edit_t_addnoise(LPIPS_th=self.args.lpips_edit_th, 
                                                            LPIPS_addnoise_th=self.args.lpips_addnoise_th,
                                                            return_clip_loss=False)


        # ----------- Get seq -----------#    
        # For editing timesteps
        if self.args.n_train_step != 0:
            seq_train = np.linspace(0, 1, self.args.n_train_step) * self.args.t_0
            seq_train = seq_train[seq_train >= self.t_edit]
            seq_train = [int(s+1e-6) for s in list(seq_train)]
            print('Uniform skip type')
        else:
            seq_train = list(range(self.t_edit, self.args.t_0))
            print('No skip')
        seq_train_next = [-1] + list(seq_train[:-1])

        # For sampling
        seq_test = np.linspace(0, 1, self.args.n_test_step) * self.args.t_0
        seq_test_edit = seq_test[seq_test >= self.t_edit]
        seq_test_edit = [int(s+1e-6) for s in list(seq_test_edit)]
        seq_test = [int(s+1e-6) for s in list(seq_test)]
        seq_test_next = [-1] + list(seq_test[:-1])


        # ----------- Model -----------#
        model = self.load_pretrained_model()
        
        # init delta_h_dict.
        delta_h_dict = {}
        for i in seq_train:
            delta_h_dict[i] = None

        if self.args.train_delta_block:
            model.setattr_layers(self.args.get_h_num)
            print("Setattr layers")
            

        model = model.to(self.device)
        model = torch.nn.DataParallel(model)


        exp_id = os.path.split(self.args.exp)[-1]
        if self.args.load_from_checkpoint:
            # load_from_checkpoint is exp_id
            save_name = f'checkpoint/{self.args.load_from_checkpoint}_LC_{self.config.data.category}_t{self.args.t_0}_ninv{self.args.n_inv_step}_ngen{self.args.n_train_step}_{self.args.n_iter - 1}.pth'
        else:
            save_name = f'checkpoint/{exp_id}_{self.args.n_iter - 1}.pth'

        if self.args.manual_checkpoint_name:
            # manual_checkpoint_name is full name of checkpoint
            save_name = 'checkpoint/' + self.args.manual_checkpoint_name
        
        elif self.args.choose_checkpoint_num:
            # choose the iter of checkpoint
            if self.args.load_from_checkpoint:
                save_name = f'checkpoint/{self.args.load_from_checkpoint}_LC_{self.config.data.category}_t{self.args.t_0}_ninv{self.args.n_inv_step}_ngen{self.args.n_train_step}_{self.args.n_iter - 1}_{self.args.choose_checkpoint_num}.pth'
            else:
                save_name = f'checkpoint/{exp_id}_{self.args.n_iter - 1}_{self.args.choose_checkpoint_num}.pth'
        
        # For global delta h
        if self.args.num_mean_of_delta_hs:
            # already exist then load
            if os.path.isfile(f"checkpoint_latent/{exp_id}_{self.args.n_test_step}_{self.args.num_mean_of_delta_hs}.pth"):
                save_name = f"checkpoint_latent/{exp_id}_{self.args.n_test_step}_{self.args.num_mean_of_delta_hs}.pth"
                load_dict = True
                delta_h_dict = {}
                for i in seq_test:
                    delta_h_dict[i] = None
            # not exist then create
            else:
                load_dict = False


        scaling_factor = self.args.n_train_step / self.args.n_test_step * self.args.hs_coeff_delta_h

        # multi attribute
        # It need to be updated multiple attr cosine & t_edit & t_addnoise
        if self.args.multiple_attr:
            multi_attr_list = self.args.multiple_attr.split(' ')
            if self.args.multiple_hs_coeff:
                multi_coeff_list = self.args.multiple_hs_coeff.split(' ')
                multi_coeff_list = [float(c) for c in multi_coeff_list]
                multi_coeff_list = multi_coeff_list + [1.0] * (len(multi_attr_list) - len(multi_coeff_list))
            else:
                multi_coeff_list = [1.0] * len(multi_attr_list)
            save_name_list = []
            max_cosine = 0
            max_attr = ""
            for attribute in multi_attr_list:
                save_name_list.append(save_name.replace('attribute', attribute))
                self.src_txts = SRC_TRG_TXT_DIC[attribute][0]
                self.trg_txts = SRC_TRG_TXT_DIC[attribute][1]
                cosine = self.set_t_edit_t_addnoise(LPIPS_th=self.args.lpips_edit_th, LPIPS_addnoise_th=self.args.lpips_addnoise_th, return_clip_loss=False)
                if cosine > max_cosine:
                    max_cosine = cosine
                    max_attr = attribute
            print(f"Max cosine: {max_cosine}, Max attribute: {max_attr}")
            self.src_txts = SRC_TRG_TXT_DIC[max_attr][0]
            self.trg_txts = SRC_TRG_TXT_DIC[max_attr][1]
            cosine = self.set_t_edit_t_addnoise(LPIPS_th=self.args.lpips_edit_th, LPIPS_addnoise_th=self.args.lpips_addnoise_th, return_clip_loss=False)

            hs_coeff = [1.0 * self.args.hs_coeff_origin_h] + [1.0 / (len(multi_attr_list))**(0.5) * scaling_factor * coeff for coeff in multi_coeff_list]
            hs_coeff = tuple(hs_coeff)

        else:
            save_name_list = [save_name]
            hs_coeff = (1.0 * self.args.hs_coeff_origin_h, 1.0 * scaling_factor)
            
    
        # Most come here
        if os.path.exists(save_name_list[0]):
            # load checkpoint
            print(f'{save_name} exists. load checkpoint')
            if self.args.train_delta_block:
                # for convince of num_mean_of_delta_hs. I've forgotten right parameters a lot of times. 
                if self.args.num_mean_of_delta_hs and load_dict:
                    self.args.train_delta_h = True
                    self.args.train_delta_block = False
                    self.args.num_mean_of_delta_hs = 0
                # delta_block load
                else:
                    for i in range(self.args.get_h_num):
                        get_h = getattr(model.module, f"layer_{i}")
                        get_h.load_state_dict(torch.load(save_name_list[i])[f"{0}"])
            
            if self.args.train_delta_h:
                saved_dict = torch.load(save_name_list[0])
                if self.args.ignore_timesteps: # global delta h is delta_h_dict[0]
                    try:
                        delta_h_dict[0] = saved_dict[f"{0}"]
                    except:
                        delta_h_dict[0] = saved_dict[0]
                else:
                    for i in delta_h_dict.keys():
                        try:
                            delta_h_dict[i] = saved_dict[f"{i}"]
                        except:
                            delta_h_dict[i] = saved_dict[i]
            
        else:
            if self.args.num_mean_of_delta_hs:
                print("There in no pre-computed mean of delta_hs! Now compute it...")
            else:
                print(f"checkpoint({save_name}) does not exist!")
                exit()

        # Scaling
        if self.args.n_train_step != self.args.n_test_step:

            if self.args.train_delta_h:
                trained_idx = 0
                test_delta_h_dict = {}
                if self.args.ignore_timesteps:
                    test_delta_h_dict[0] = delta_h_dict[0]
                interval_seq = (seq_train[1] - seq_train[0])

                if not load_dict:
                
                    for i in seq_test_edit:
                        test_delta_h_dict[i] = delta_h_dict[seq_train[trained_idx]]
                        
                        if i > seq_train[trained_idx] - interval_seq:
                            if trained_idx < len(seq_train) - 1:
                                trained_idx += 1                        
                    
                    del delta_h_dict
                    delta_h_dict = test_delta_h_dict
            
            else:
                for i in seq_test:
                    delta_h_dict[i] = None
        
        # For interpolation
        if self.args.delta_interpolation:
            if self.args.multiple_attr:
                assert self.args.get_h_num == 2, "delta_multiple_attr_interpolation is only supported for get_h_num == 2"
                interpolation_vals = np.linspace(self.args.min_delta, self.args.max_delta, self.args.num_delta)
                interpolation_vals = interpolation_vals.tolist()
                hs_coeff = list(hs_coeff)

                hs_coeff_list = []

                for val_1 in interpolation_vals:
                    for val_2 in interpolation_vals:
                        coeff_tuple = (1.0, val_1*hs_coeff[1], val_2*hs_coeff[2])
                        hs_coeff_list.append(coeff_tuple)

                del hs_coeff
                hs_coeff = hs_coeff_list

            else:
                interpolation_vals = np.linspace(self.args.min_delta, self.args.max_delta, self.args.num_delta)
                interpolation_vals = interpolation_vals.tolist()
                
                hs_coeff_list = []

                for val in interpolation_vals:
                    coeff_tuple = [val*elem for elem in hs_coeff]
                    coeff_tuple[0] = 1.0
                    hs_coeff_list.append(tuple(coeff_tuple))
                
                del hs_coeff
                hs_coeff = hs_coeff_list
        
        if self.args.num_mean_of_delta_hs:
            assert self.args.bs_train == 1, "if you want to use mean, batch_size must be 1"

        # ----------- Pre-compute -----------#
        print("Prepare identity latent...")
        # get xT
        if self.args.load_random_noise:
            img_lat_pairs_dic = self.random_noise_pairs(model, saved_noise=self.args.saved_random_noise, save_imgs=self.args.save_precomputed_images)
        else:
            img_lat_pairs_dic = self.precompute_pairs(model, self.args.save_precomputed_images)
        
        if self.args.target_image_id:
            self.args.target_image_id = self.args.target_image_id.split(" ")
            self.args.target_image_id = [int(i) for i in self.args.target_image_id]

        # Unfortunately, ima_lat_pairs_dic does not match with batch_size
        x_lat_tensor = None
        x0_tensor = None
        model.eval()

        # Train set
        if self.args.do_train:
            for step, (x0, _, x_lat) in enumerate(img_lat_pairs_dic['train']):

                if self.args.target_image_id:
                    assert self.args.bs_train == 1, "target_image_id is only supported for batch_size == 1"
                    if not step in self.args.target_image_id:
                        continue
                if self.args.start_image_id > step:
                    continue


                if x_lat_tensor is None:
                    x_lat_tensor = x_lat
                    if self.args.use_x0_tensor:
                                x0_tensor = x0
                else:
                    x_lat_tensor = torch.cat((x_lat_tensor, x_lat), dim=0)
                    if self.args.use_x0_tensor:
                                x0_tensor = torch.cat((x0_tensor, x0), dim=0)
                if (step+1) % self.args.bs_train != 0:
                    continue

                self.save_image(model, x_lat_tensor, seq_test, seq_test_next,
                                            save_x0 = self.args.save_x0, save_x_origin = self.args.save_x_origin,
                                            x0_tensor=x0_tensor, delta_h_dict=delta_h_dict,
                                            folder_dir=self.args.test_image_folder, get_delta_hs=self.args.num_mean_of_delta_hs,
                                            save_process_origin=self.args.save_process_origin, save_process_delta_h=self.args.save_process_delta_h,
                                            file_name=f'train_{step}_{self.args.n_iter - 1}', hs_coeff=hs_coeff,
                                            )
                                        
                if step == self.args.n_train_img - 1:
                    break
                # if mean_of_delta_hs is not exist,
                if step == self.args.num_mean_of_delta_hs -1:
                    for keys in delta_h_dict.keys():
                        if delta_h_dict[keys] is None:
                            continue
                        delta_h_dict[keys] = delta_h_dict[keys]/(step+1)
                    
                    sumation_delta_h = None
                    sumation_num = 0
                    for keys in delta_h_dict.keys():
                        if sumation_delta_h is None:
                            sumation_delta_h = copy.deepcopy(delta_h_dict[keys])
                            sumation_num = 1
                        else:
                            if delta_h_dict[keys] is None:
                                continue
                            sumation_delta_h += delta_h_dict[keys]
                            sumation_num += 1
                    # if ignore_timesteps, only use delta_h_dict[0]
                    delta_h_dict[0] = sumation_delta_h/sumation_num

                    torch.save(delta_h_dict, f'checkpoint_latent/{exp_id}_{self.args.n_test_step}_{self.args.num_mean_of_delta_hs}.pth')
                    print(f'Dict: checkpoint_latent/{exp_id}_{self.args.n_test_step}_{self.args.num_mean_of_delta_hs}.pth is saved.')

                    self.args.num_mean_of_delta_hs = 0
                    print("now we use mean of delta_hs")
 
                x_lat_tensor = None
        
        # Test set
        if self.args.do_test:
            x_lat_tensor = None

            for step, (x0, _, x_lat) in enumerate(img_lat_pairs_dic['test']):

                if self.args.target_image_id:
                    assert self.args.bs_train == 1, "target_image_id is only supported for batch_size == 1"
                    if not step in self.args.target_image_id:
                        continue

                if self.args.start_image_id > step:
                    continue

                if x_lat_tensor is None:
                    x_lat_tensor = x_lat
                    if self.args.use_x0_tensor:
                                x0_tensor = x0
                else:
                    x_lat_tensor = torch.cat((x_lat_tensor, x_lat), dim=0)
                    if self.args.use_x0_tensor:
                                x0_tensor = torch.cat((x0_tensor, x0), dim=0)
                if (step+1) % self.args.bs_train != 0:
                    continue

                self.save_image(model, x_lat_tensor, seq_test, seq_test_next,
                                            save_x0 = self.args.save_x0, save_x_origin = self.args.save_x_origin,
                                            x0_tensor=x0_tensor, delta_h_dict=delta_h_dict,
                                            folder_dir=self.args.test_image_folder, get_delta_hs=self.args.num_mean_of_delta_hs,
                                            save_process_origin=self.args.save_process_origin, save_process_delta_h=self.args.save_process_delta_h,
                                            file_name=f'test_{step}_{self.args.n_iter - 1}', hs_coeff=hs_coeff,
                                            )
                                        
                if step == self.args.n_test_img - 1:
                    break
                x_lat_tensor = None


    @torch.no_grad()
    def precompute_pairs_with_h(self, model, img_path):


        if not os.path.exists('./precomputed'):
            os.mkdir('./precomputed')

        save_path = "_".join(img_path.split(".")[-2].split('/')[-2:])
        save_path = self.config.data.category + '_inv' + str(self.args.n_inv_step) + '_' + save_path + '.pt'
        save_path = os.path.join('precomputed', save_path)

        n = 1

        print("Precompute multiple h and x_T")
        seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0
        seq_inv = [int(s+1e-6) for s in list(seq_inv)]
        seq_inv_next = [-1] + list(seq_inv[:-1])

        if os.path.exists(save_path):
            print("Precomputed pairs already exist")
            img_lat_pair = torch.load(save_path)
            return img_lat_pair
        else:
            tmp_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            
            image = Image.open(img_path).convert('RGB')

            width, height = image.size
            if width > height:
                image = transforms.CenterCrop(height)(image)
            else:
                image = transforms.CenterCrop(width)(image)
            
            image = tmp_transform(image)

            h_dic = {}

            x0 = image.unsqueeze(0).to(self.device)

            x = x0.clone()
            model.eval()
            time_s = time.time()

            with torch.no_grad():
                with tqdm(total=len(seq_inv), desc=f"Inversion processing") as progress_bar:
                    for it, (i, j) in enumerate(zip((seq_inv_next[1:]), (seq_inv[1:]))):
                        t = (torch.ones(n) * i).to(self.device)
                        t_prev = (torch.ones(n) * j).to(self.device)

                        x, _, _, h = denoising_step(x, t=t, t_next=t_prev, models=model,
                                            logvars=self.logvar,
                                            sampling_type='ddim',
                                            b=self.betas,
                                            eta=0,
                                            learn_sigma=self.learn_sigma,
                                            )
                        progress_bar.update(1)
                        h_dic[i] = h.detach().clone().cpu()
                        

                time_e = time.time()
                progress_bar.set_description(f"Inversion processing time: {time_e - time_s:.2f}s")
                x_lat = x.clone()
            print("Generative process is skipped")

            img_lat_pairs = [x0, 0 , x_lat.detach().clone().cpu(), h_dic]
            
            torch.save(img_lat_pairs,save_path)
            print("Precomputed pairs are saved to ", save_path)

            return img_lat_pairs


    # ----------- Pre-compute -----------#
    @torch.no_grad()
    def precompute_pairs(self, model, save_imgs=False):
    
        print("Prepare identity latent")
        seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0
        seq_inv = [int(s+1e-6) for s in list(seq_inv)]
        seq_inv_next = [-1] + list(seq_inv[:-1])

        n = 1
        img_lat_pairs_dic = {}

        for mode in ['train', 'test']:
            img_lat_pairs = []
            if self.config.data.dataset == "IMAGENET":
                if self.args.target_class_num is not None:
                    pairs_path = os.path.join('precomputed/',
                                              f'{self.config.data.category}_{IMAGENET_DIC[str(self.args.target_class_num)][1]}_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth')
                else:
                    pairs_path = os.path.join('precomputed/',
                                              f'{self.config.data.category}_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth')

            else:
                if mode == 'train':
                    pairs_path = os.path.join('precomputed/',
                                          f'{self.config.data.category}_{mode}_t{self.args.t_0}_nim{self.args.n_train_img}_ninv{self.args.n_inv_step}_pairs.pth')
                else:
                    pairs_path = os.path.join('precomputed/',
                                              f'{self.config.data.category}_{mode}_t{self.args.t_0}_nim{self.args.n_test_img}_ninv{self.args.n_inv_step}_pairs.pth')
            print(pairs_path)
            if os.path.exists(pairs_path) and not self.args.re_precompute:
                print(f'{mode} pairs exists')
                img_lat_pairs_dic[mode] = torch.load(pairs_path, map_location=torch.device('cpu'))
                if save_imgs:
                    for step, (x0, x_id, x_lat) in enumerate(img_lat_pairs_dic[mode]):
                        tvu.save_image((x0 + 1) * 0.5, os.path.join(self.args.image_folder, f'{mode}_{step}_0_orig.png'))
                        tvu.save_image((x_id + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                    f'{mode}_{step}_1_rec_ninv{self.args.n_inv_step}.png'))
                        if step == self.args.n_precomp_img - 1:
                            break
                continue
            else:

                exist_num = 0
                for exist_precompute_num in reversed(range(self.args.n_train_img if mode == 'train' else self.args.n_test_img)):
                    tmp_path = os.path.join('precomputed/',
                                          f'{self.config.data.category}_{mode}_t{self.args.t_0}_nim{exist_precompute_num}_ninv{self.args.n_inv_step}_pairs.pth')
                    if os.path.exists(tmp_path):
                        print(f'latest {mode} pairs are exist. Continue precomputing...')
                        img_lat_pairs = img_lat_pairs + torch.load(tmp_path, map_location=torch.device('cpu'))
                        exist_num = exist_precompute_num
                        break

                if self.config.data.category == 'CUSTOM':
                    DATASET_PATHS["custom_train"] = self.args.custom_train_dataset_dir
                    DATASET_PATHS["custom_test"] = self.args.custom_test_dataset_dir

                train_dataset, test_dataset = get_dataset(self.config.data.dataset, DATASET_PATHS, self.config,
                                                              target_class_num=self.args.target_class_num)

                loader_dic = get_dataloader(train_dataset, test_dataset, bs_train=1,#self.args.bs_train,
                                            num_workers=self.config.data.num_workers, shuffle=self.args.shuffle_train_dataloader)
                loader = loader_dic[mode]

                if self.args.save_process_origin:
                    save_process_folder = os.path.join(self.args.image_folder, f'inversion_process')
                    if not os.path.exists(save_process_folder):
                        os.makedirs(save_process_folder)

            for step, img in enumerate(loader):
                if (mode == "train" and step == self.args.n_train_img) or (mode == "test" and step == self.args.n_test_img):
                    break
                if exist_num != 0:
                    exist_num = exist_num - 1
                    continue
                x0 = img.to(self.config.device)
                if save_imgs:
                    tvu.save_image((x0 + 1) * 0.5, os.path.join(self.args.image_folder, f'{mode}_{step}_0_orig.png'))

                x = x0.clone()
                model.eval()
                time_s = time.time()
                with torch.no_grad():
                    with tqdm(total=len(seq_inv), desc=f"Inversion process {mode} {step}") as progress_bar:
                        for it, (i, j) in enumerate(zip((seq_inv_next[1:]), (seq_inv[1:]))):
                            t = (torch.ones(n) * i).to(self.device)
                            t_prev = (torch.ones(n) * j).to(self.device)

                            x, _, _, _ = denoising_step(x, t=t, t_next=t_prev, models=model,
                                               logvars=self.logvar,
                                               sampling_type='ddim',
                                               b=self.betas,
                                               eta=0,
                                               learn_sigma=self.learn_sigma,
                                               )
                            progress_bar.update(1)
                    
                    time_e = time.time()
                    print(f'{time_e - time_s} seconds')
                    x_lat = x.clone()
                    if save_imgs:
                        tvu.save_image((x_lat + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                    f'{mode}_{step}_1_lat_ninv{self.args.n_inv_step}.png'))

                    with tqdm(total=len(seq_inv), desc=f"Generative process {mode} {step}") as progress_bar:
                        time_s = time.time()
                        for it, (i, j) in enumerate(zip(reversed((seq_inv)), reversed((seq_inv_next)))):
                            t = (torch.ones(n) * i).to(self.device)
                            t_next = (torch.ones(n) * j).to(self.device)

                            x, x0t, _, _ = denoising_step(x, t=t, t_next=t_next, models=model,
                                               logvars=self.logvar,
                                               sampling_type=self.args.sample_type,
                                               b=self.betas,
                                               learn_sigma=self.learn_sigma)
                            progress_bar.update(1)
                            if self.args.save_process_origin:
                                tvu.save_image((x + 1) * 0.5, os.path.join(save_process_folder, f'xt_{step}_{it}_{t[0]}.png'))
                                tvu.save_image((x0t + 1) * 0.5, os.path.join(save_process_folder, f'x0t_{step}_{it}_{t[0]}.png'))
                        time_e = time.time()
                        print(f'{time_e - time_s} seconds')

                    img_lat_pairs.append([x0, x.detach().clone(), x_lat.detach().clone()])
                
                if save_imgs:
                    tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                            f'{mode}_{step}_1_rec_ninv{self.args.n_inv_step}.png'))
                

            img_lat_pairs_dic[mode] = img_lat_pairs
            # pairs_path = os.path.join('precomputed/',
            #                           f'{self.config.data.category}_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth')
            torch.save(img_lat_pairs, pairs_path)

        return img_lat_pairs_dic

    # ----------- Get random latent -----------#
    @torch.no_grad()
    def random_noise_pairs(self, model, saved_noise=False, save_imgs=False):

        print("Prepare random latent")
        seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0
        seq_inv = [int(s+1e-6) for s in list(seq_inv)]
        seq_inv_next = [-1] + list(seq_inv[:-1])

        n = 1
        img_lat_pairs_dic = {}

        if saved_noise:

            for mode in ['train', 'test']:
                img_lat_pairs = []
                if self.config.data.dataset == "IMAGENET":
                    if self.args.target_class_num is not None:
                        pairs_path = os.path.join('precomputed/',
                                                f'{self.config.data.category}_{IMAGENET_DIC[str(self.args.target_class_num)][1]}_{mode}_random_noise_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth')
                    else:
                        pairs_path = os.path.join('precomputed/',
                                                f'{self.config.data.category}_{mode}_random_noise_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth')

                else:
                    if mode == 'train':
                        pairs_path = os.path.join('precomputed/',
                                            f'{self.config.data.category}_{mode}_random_noise_nim{self.args.n_train_img}_ninv{self.args.n_inv_step}_pairs.pth')
                    else:
                        pairs_path = os.path.join('precomputed/',
                                                f'{self.config.data.category}_{mode}_random_noise_nim{self.args.n_test_img}_ninv{self.args.n_inv_step}_pairs.pth')
                print(pairs_path)
                if os.path.exists(pairs_path):
                    print(f'{mode} pairs exists')
                    img_lat_pairs_dic[mode] = torch.load(pairs_path, map_location=torch.device('cpu'))
                    if save_imgs:
                        for step, (_, x_id, x_lat) in enumerate(img_lat_pairs_dic[mode]):
                            tvu.save_image((x_id + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                        f'{mode}_{step}_1_rec_ninv{self.args.n_inv_step}.png'))
                            if step == self.args.n_precomp_img - 1:
                                break
                    continue
                
                step = 0
                while True:
                    
                    with torch.no_grad():
                        x_lat = torch.randn((1, self.config.data.channels, self.config.data.image_size, self.config.data.image_size)).to(self.device)

                        if save_imgs:
                            tvu.save_image((x_lat + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                        f'{mode}_{step}_1_lat_ninv{self.args.n_inv_step}.png'))

                        with tqdm(total=len(seq_inv), desc=f"Generative process {mode} {step}") as progress_bar:
                            time_s = time.time()
                            x = x_lat
                            for it, (i, j) in enumerate(zip(reversed((seq_inv)), reversed((seq_inv_next)))):
                                t = (torch.ones(n) * i).to(self.device)
                                t_next = (torch.ones(n) * j).to(self.device)

                                x, _, _, _ = denoising_step(x, t=t, t_next=t_next, models=model,
                                                logvars=self.logvar,
                                                sampling_type=self.args.sample_type,
                                                b=self.betas,
                                                learn_sigma=self.learn_sigma)
                                progress_bar.update(1)
                            time_e = time.time()
                            print(f'{time_e - time_s} seconds')
                        # img_lat_pairs.append([None, x.detach().clone(), x_lat.detach().clone()])
                        img_lat_pairs.append([x.detach().clone(), x.detach().clone(), x_lat.detach().clone()])

                    

                    if save_imgs:
                        tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                f'{mode}_{step}_1_rec_ninv{self.args.n_inv_step}.png'))
                    if (mode == "train" and step == self.args.n_train_img - 1) or (mode == "test" and step == self.args.n_test_img - 1):
                        break
                    step += 1

                img_lat_pairs_dic[mode] = img_lat_pairs
                torch.save(img_lat_pairs, pairs_path)

        else:
            train_lat = []
            for i in range(self.args.n_train_img):
                lat = torch.randn((1, self.config.data.channels, self.config.data.image_size, self.config.data.image_size)).to(self.device)
                # train_lat.append([None, None, lat])
                train_lat.append([torch.zeros_like(lat), torch.zeros_like(lat), lat])

            img_lat_pairs_dic['train'] = train_lat

            test_lat = []
            for i in range(self.args.n_test_img):
                lat = torch.randn((1, self.config.data.channels, self.config.data.image_size, self.config.data.image_size)).to(self.device)
                # test_lat.append([None, None, lat])
                test_lat.append([torch.zeros_like(lat), torch.zeros_like(lat), lat])

            img_lat_pairs_dic['test'] = test_lat

            

        return img_lat_pairs_dic

    @torch.no_grad()
    def compute_lpips_distance(self):
        import pickle
        print("Get lpips distance...")
        self.args.bs_train = 1

        # ----------- Model -----------#

        model = self.load_pretrained_model()

        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        import lpips

        loss_fn_alex = lpips.LPIPS(net='alex')
        loss_fn_alex = loss_fn_alex.to(self.device)    
        

        # ----------- Pre-compute -----------#
        print("Prepare identity latent")
        seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0
        seq_inv = [int(s+1e-6) for s in list(seq_inv)]
        seq_inv_next = [-1] + list(seq_inv[:-1])

        print("the list is Unique? :", len(seq_inv) == len(set(seq_inv)))

        train_dataset, test_dataset = get_dataset(self.config.data.dataset, DATASET_PATHS, self.config,
                                                        target_class_num=self.args.target_class_num)

        loader_dic = get_dataloader(train_dataset, test_dataset, bs_train=1,#self.args.bs_train,
                                    num_workers=self.config.data.num_workers)
        loader = loader_dic["train"]
        print("Load dataset done")

        lpips_distance_list = {}
        lpips_distance_list_x0_t = {}
        for seq in seq_inv[1:]:
            lpips_distance_list[seq] = []
            lpips_distance_list_x0_t[seq] = []

        lpips_distance_std_list = {}
        lpips_distance_std_list_x0_t = {}
        for seq in seq_inv[1:]:
            lpips_distance_std_list[seq] = []
            lpips_distance_std_list_x0_t[seq] = []

        save_imgs = True

        for step, img in enumerate(loader):
            x0 = img.to(self.device)
            if save_imgs:
                tvu.save_image((x0 + 1) * 0.5, os.path.join(self.args.image_folder, f'LPIPS_{step}_0_orig.png'))

            x = x0.clone()
            model.eval()
            time_s = time.time()
            with torch.no_grad():
                with tqdm(total=len(seq_inv), desc=f"Inversion process {step}") as progress_bar:
                    for it, (i, j) in enumerate(zip((seq_inv_next[1:]), (seq_inv[1:]))):
                        t = (torch.ones(self.args.bs_train) * i).to(self.device)
                        t_prev = (torch.ones(self.args.bs_train) * j).to(self.device)

                        x, x0_t, _, _ = denoising_step(x, t=t, t_next=t_prev, models=model,
                                            logvars=self.logvar,
                                            sampling_type='ddim',
                                            b=self.betas,
                                            eta=0,
                                            learn_sigma=self.learn_sigma,
                                            )
                        lpips_x = loss_fn_alex(x, x0)
                        lpips_x0 = loss_fn_alex(x0_t, x0)
                        lpips_distance_list[j].append(lpips_x.item())
                        lpips_distance_list_x0_t[j].append(lpips_x0.item())
                        if save_imgs:
                            tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                f'LPIPS_{step}_{j}.png'))
                            tvu.save_image((x0_t + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                f'X0_t_LPIPS_{step}_{j}.png'))
                        progress_bar.update(1)
                
                time_e = time.time()
                print(f'{time_e - time_s} seconds')
            
            save_imgs = False
            if self.args.n_train_img == step:
                break
        
        result_x_tsv = ""
        result_x_std_tsv = ""
        result_x0_tsv = ""
        result_x0_std_tsv = ""
        for seq in seq_inv[1:]:
            lpips_distance_std_list[seq] = np.std(lpips_distance_list[seq])
            lpips_distance_list[seq] = np.mean(lpips_distance_list[seq])
            
            # print(f"{seq} : {lpips_distance_list[seq]}")
            lpips_distance_std_list_x0_t[seq] = np.std(lpips_distance_list_x0_t[seq])
            lpips_distance_list_x0_t[seq] = np.mean(lpips_distance_list_x0_t[seq])
            
            # print(f"{seq} : {lpips_distance_list_x0_t[seq]}")
            result_x_tsv += f"{seq}\t{lpips_distance_list[seq]}\n"
            result_x_std_tsv += f"{seq}\t{lpips_distance_std_list[seq]}\n"
            result_x0_tsv += f"{seq}\t{lpips_distance_list_x0_t[seq]}\n"
            result_x0_std_tsv += f"{seq}\t{lpips_distance_std_list_x0_t[seq]}\n"

        with open(os.path.join("utils", f"{(self.args.config).split('.')[0]}_LPIPS_distance_x.tsv"), "w") as f:
            f.write(result_x_tsv)
        with open(os.path.join("utils", f"{(self.args.config).split('.')[0]}_LPIPS_distance_x_std.tsv"), "w") as f:
            f.write(result_x_std_tsv)
        with open(os.path.join("utils", f"{(self.args.config).split('.')[0]}_LPIPS_distance_x0_t.tsv"), "w") as f:
            f.write(result_x0_tsv)
        with open(os.path.join("utils", f"{(self.args.config).split('.')[0]}_LPIPS_distance_x0_t_std.tsv"), "w") as f:
            f.write(result_x0_std_tsv)



    @torch.no_grad()
    def set_t_edit_t_addnoise(self, LPIPS_th=0.33, LPIPS_addnoise_th=0.1, return_clip_loss=False):

        clip_loss_func = CLIPLoss(
            self.device,
            lambda_direction=1,
            lambda_patch=0,
            lambda_global=0,
            lambda_manifold=0,
            lambda_texture=0,
            clip_model=self.args.clip_model_name)

        # ----------- Get clip cosine similarity -----------#
        print("Texts:", self.src_txts, self.trg_txts)
        scr_token = clip_loss_func.tokenize(self.src_txts)
        trg_token = clip_loss_func.tokenize(self.trg_txts)
        text_feature_scr = clip_loss_func.encode_text(scr_token)
        text_feature_trg = clip_loss_func.encode_text(trg_token)

        ## get cosine distance between features
        text_cos_distance = torch.nn.CosineSimilarity(dim=1, eps=1e-6)(text_feature_scr, text_feature_trg)
        print("text_cos_distance", text_cos_distance.item())
        cosine = text_cos_distance.item()
        
        # t_edit is from LPIPS(x0_t, x0)
        print("get t_edit from LPIPS distance!")
        # LPIPS_th = 0.33
        LPIPS_th = LPIPS_th * cosine

        dataset_name = str(self.args.config).split(".")[0]
        if dataset_name == "custom":
            dataset_name = self.args.custom_dataset_name
        LPIPS_file_name = f"{dataset_name}_LPIPS_distance_x0_t.tsv"
        LPIPS_file_path = os.path.join("utils", LPIPS_file_name)
        if not os.path.exists(LPIPS_file_path):
            if (self.args.user_defined_t_edit and self.args.user_defined_t_addnoise):
                self.t_edit = self.args.user_defined_t_edit
                self.t_addnoise = self.args.user_defined_t_addnoise
                print("user_defined t_edit and t_addnoise")
                print(f"t_edit: {self.t_edit}")
                print(f"t_addnoise: {self.t_addnoise}")
                if return_clip_loss:
                    return cosine, clip_loss_func
                else:
                    return cosine
            else:
                print(f"LPIPS file not found, get LPIPS distance first!  : {LPIPS_file_path}")
                raise ValueError
        import csv
        lpips_dict = {}
        with open(LPIPS_file_path, "r") as f:
            lines = csv.reader(f, delimiter="\t")
            for line in lines:
                lpips_dict[int(line[0])] = float(line[1])

        sorted_lpips_dict_key_list = list(lpips_dict.keys())
        sorted_lpips_dict_key_list.sort()
        if len(sorted_lpips_dict_key_list) != 1000:
            # even if not fully steps, it's okay.
            print("Warning: LPIPS file not fully steps! (But it's okay. lol)")
        
        if self.args.user_defined_t_edit:
            # when you use user_defined_t_edit but not user_defined_t_addnoise
            t_edit = self.args.user_defined_t_edit
        else:
            # get t_edit
            for key in sorted_lpips_dict_key_list:
                if lpips_dict[key] >= LPIPS_th:
                    t_edit = key
                    break

        self.t_edit = t_edit
        print(f"t_edit: {self.t_edit}")

        # t_boost is from LPIPS(xt, x0)
        if self.args.user_defined_t_addnoise:
            # when you use user_defined_t_addnoise but not user_defined_t_edit
            t_addnoise = self.args.user_defined_t_addnoise
        else:
            if self.args.add_noise_from_xt:
                LPIPS_file_name = f"{dataset_name}_LPIPS_distance_x.tsv"
                LPIPS_file_path = os.path.join("utils", LPIPS_file_name)
                if not os.path.exists(LPIPS_file_path):
                    print("LPIPS file not found, get LPIPS distance first!")
                    raise ValueError
                lpips_dict = {}
                with open(LPIPS_file_path, "r") as f:
                    lines = csv.reader(f, delimiter="\t")
                    for line in lines:
                        lpips_dict[int(line[0])] = float(line[1])

                sorted_lpips_dict_key_list = list(lpips_dict.keys())
                sorted_lpips_dict_key_list.sort()
            
            # get t_add_noise
            for key in sorted_lpips_dict_key_list:
                if lpips_dict[key] >= LPIPS_addnoise_th:
                    t_addnoise = key
                    break
        self.t_addnoise = t_addnoise
        print(f"t_addnoise: {self.t_addnoise}")

        if return_clip_loss:
            return cosine, clip_loss_func
        else:
            return cosine

    


