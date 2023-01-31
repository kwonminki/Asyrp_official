
# Diffusion Models already have a Semantic Latent Space (ICLR2023 notable-top-25%)

[![arXiv](https://img.shields.io/badge/arXiv-2110.02711-red)](https://arxiv.org/abs/2210.10960) [![project_page](https://img.shields.io/badge/project_page-orange)](https://kwonminki.github.io/Asyrp/)


> **Diffusion Models already have a Semantic Latent Space**<br>
> [Mingi Kwon](https://drive.google.com/file/d/1d1TOCA20KmYnY8RvBvhFwku7QaaWIMZL/view?usp=share_link), [Jaeseok Jeong](https://drive.google.com/file/d/14uHCJLoR1AFydqV_neGjl1H2rjN4HBdv/view), [Youngjung Uh](https://vilab.yonsei.ac.kr/member/professor) <br>
> Arxiv preprint.
> 
>**Abstract**: <br>
Diffusion models achieve outstanding generative performance in various domains. Despite their great success, they lack semantic latent space which is essential for controlling the generative process. To address the problem, we propose asymmetric reverse process (Asyrp) which discovers the semantic latent space in frozen pretrained diffusion models. Our semantic latent space, named h-space, has nice properties for accommodating semantic image manipulation: homogeneity, linearity, robustness, and consistency across timesteps. In addition, we introduce a principled design of the generative process for versatile editing and quality boosting by quantifiable measures: editing strength of an interval and quality deficiency at a timestep. Our method is applicable to various architectures (DDPM++, iDDPM, and ADM) and datasets (CelebA-HQ, AFHQ-dog, LSUN-church, LSUN-bedroom, and METFACES).
 

## Description
This repo includes the official Pytorch implementation of **Asyrp**: Diffusion Models already have a Semantic Latent Space.

- **Asyrp** allows using *h-space*, the bottleneck of the U-Net, as a semantic latent space of diffusion models.

![image](https://user-images.githubusercontent.com/33779055/210209549-500e57d1-0a38-4167-a437-f1dcc44b5a5a.png) ![image](https://user-images.githubusercontent.com/33779055/210209586-096ec082-f2d2-4690-84c9-ce0143361069.png) ![image](https://user-images.githubusercontent.com/33779055/210209619-6091bf02-e81b-468f-a2d0-df893040ab66.png)

Edited real images (Top) as `Happy dog` (Bottom). So cute!!





## Getting Started
We recommend running our code using NVIDIA GPU + CUDA, CuDNN.

### Pretrained Models for Asyrp
Asyrp works on the checkpoints of pretrained diffusion models.


| Image Type to Edit |Size| Pretrained Model | Dataset | Reference Repo. 
|---|---|---|---|---
| Human face |256×256| Diffusion (Auto) | [CelebA-HQ](https://arxiv.org/abs/1710.10196) | [SDEdit](https://github.com/ermongroup/SDEdit)
| Human face |256×256| [Diffusion](https://1drv.ms/u/s!AkQjJhxDm0Fyhqp_4gkYjwVRBe8V_w?e=Et3ITH) | [CelebA-HQ](https://arxiv.org/abs/1710.10196) | [P2 weighting](https://github.com/jychoi118/P2-weighting)
| Human face |256×256| [Diffusion](https://1drv.ms/u/s!AkQjJhxDm0Fyhqp_4gkYjwVRBe8V_w?e=Et3ITH) | [FFHQ](https://arxiv.org/abs/1812.04948) | [P2 weighting](https://github.com/jychoi118/P2-weighting)
| Church |256×256| Diffusion (Auto) | [LSUN-Bedroom](https://www.yf.io/p/lsun) | [SDEdit](https://github.com/ermongroup/SDEdit) 
| Bedroom |256×256| Diffusion (Auto) | [LSUN-Church](https://www.yf.io/p/lsun) | [SDEdit](https://github.com/ermongroup/SDEdit) 
| Dog face |256×256| [Diffusion](https://1drv.ms/u/s!AkQjJhxDm0Fyhqp_4gkYjwVRBe8V_w?e=Et3ITH) | [AFHQ-Dog](https://arxiv.org/abs/1912.01865) | [ILVR](https://github.com/jychoi118/ilvr_adm)
| Painting face |256×256| [Diffusion](https://1drv.ms/u/s!AkQjJhxDm0Fyhqp_4gkYjwVRBe8V_w?e=Et3ITH) | [METFACES](https://arxiv.org/abs/2006.06676) | [P2 weighting](https://github.com/jychoi118/P2-weighting)
| ImageNet |256x256| [Diffusion](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt) | [ImageNet](https://image-net.org/index.php) | [Guided Diffusion](https://github.com/openai/guided-diffusion)

- The pretrained Diffuson models on 256x256 images in [CelebA-HQ](https://arxiv.org/abs/1710.10196), [LSUN-Church](https://www.yf.io/p/lsun), and [LSUN-Bedroom](https://www.yf.io/p/lsun) are automatically downloaded in the code. (codes from [DiffusionCLIP](https://github.com/gwang-kim/DiffusionCLIP))
- In contrast, you need to download the models pretrained on other datasets in the table and put it in the `./pretrained` directory. 
- You can manually revise the checkpoint paths and names in the `./configs/paths_config.py` file.

- We used CelebA-HQ pretrained model from SDEdit but we found from P2 weighting is better. **We highly recommend to use P2 weighting models rather than SDEdit.**

### Datasets 
To precompute latents and find the direction of *h-space*, you need about 100+ images in the dataset. You can use both **sampled images** from the pretrained models or **real images** from the pretraining dataset. 

If you want to use **real images**, check the URLs :
- [CelebA-HQ](https://drive.google.com/drive/folders/0B4qLcYyJmiz0TXY1NG02bzZVRGs?resourcekey=0-arAVTUfW9KRhN-irJchVKQ), [AFHQ-Dog](https://github.com/clovaai/stargan-v2), [LSUN-Church](https://www.yf.io/p/lsun), [LSUN-Bedroom](https://www.yf.io/p/lsun), [ImageNet](https://image-net.org/index.php), [METFACES](https://github.com/NVlabs/metfaces-dataset), [FFHQ](https://github.com/NVlabs/ffhq-dataset)

You can simply modify `./configs/paths_config.py` for dataset path.

### CUSTOM Datasets
If you want to use a custom dataset, you can use the `config/custom.yml` file.
- You have to match `data.dataset` in `custom.yml` with your data domain. For example, if you want to use Human Face images, `data.dataset` should be `CelebA_HQ` or `FFHQ`. 
- `data.category` should be `'CUSTOM'`
- Then, you can use the below arguments:
```
--custom_train_dataset_dir "your/costom/dataset/dir/train"    \
--custom_test_dataset_dir "your/costom/dataset/dir/test"      \
```

### Get LPIPS distance
We provide precomputed LPIPS distances for `CelebA_HQ`, `LSUN-Bedroom`, `LSUN-Church`, `AFHQ-Dog`, and `METFACES` in the `./utils`.

If you want to use the custom/other dataset, we recommand to precompute LPIPS distance.

To precompute LPIPS distance for automatically defined t_edit & t_boost, run the following commands using `script_get_lpips.sh`.
```
python main.py  --lpips                  \
                --config $config         \
                --exp ./runs/tmp         \
                --edit_attr test         \
                --n_train_img 100        \
                --n_inv_step 1000   
```
- `$config` : `celeba.yml` for human face, `bedroom.yml` for bedroom, `church.yml` for church, `afhq.yml` for dog face, `imagenet.yml` for images from ImageNet, `metface.yml` for artistic face from METFACES, `ffqh.yml` for human face from FFHQ.
- `exp`: Experiment name.
- `edit_attr`: Attribute to edit. But not used for now. you can use `./utils/text_dic.py` to predefined source-target text pairs or define new pair. 
- `n_train_img` : LPIPS distance from # of images.
- `n_inv_step` : # of steps during the generative pross for the inversion. You can use `--n_inv_step 50` for speed. 


## Asyrp
To train the implicit function f, you can prepare two optional things. 1) get LPIPS distances 2) precompute

We alredy provide precomputed LPIPS distances for `CelebA_HQ`, `LSUN-Bedroom`, `LSUN-Church`, `AFHQ-Dog`, and `METFACES` in the `./utils`.

If you want to use your own defined-t_edit (e.g., 500) and defined-t_boost (e.g., 200), you don't need to get LPIPS distances.

For that case, you can can use the below arguments:
```
--user_defined_t_edit 500       \
--user_defined_t_addnoise 200   \
```

If you want to train with sampled images, you don't need to precompute real images.
For that case you can use the below argument:
```
--load_random_noise
```

### Precompute real images
To precompute real images for saving time, run the follwing commands using `script_precompute.sh`.
```
python main.py  --run_train          \
                --config $config     \
                --exp ./runs/tmp     \
                --edit_attr test     \
                --do_train 1         \
                --do_test 1          \
                --n_train_img 100    \
                --n_test_img 32      \
                --bs_train 1         \
                --n_inv_step 50      \
                --n_train_step 50    \
                --n_test_step 50     \
                --just_precompute    
```

### Train the implicit function
To train the implicit function, run the following commands using `script_train.sh`
```
python main.py  --run_train                    \
                --config $config               \
                --exp ./runs/example           \
                --edit_attr $guid              \
                --do_train 1                   \
                --do_test 1                    \
                --n_train_img 100              \
                --n_test_img 32                \
                --n_iter 5                     \
                --bs_train 1                   \
                --t_0 999                      \
                --n_inv_step 50                \
                --n_train_step 50              \
                --n_test_step 50               \
                --get_h_num 1                  \
                --train_delta_block            \
                --save_x0                      \
                --use_x0_tensor                \
                --lr_training 0.5              \
                --clip_loss_w 1.0              \
                --l1_loss_w 3.0                \
                --add_noise_from_xt            \
                --lpips_addnoise_th 1.2        \
                --lpips_edit_th 0.33           \
                --sh_file_name $sh_file_name   \

                (optional - if you pass "get LPIPS")
                --user_defined_t_edit 500      \
                --user_defined_t_addnoise 200  \

                (optional - if you pass "precompute")
                --load_random_noise
```
- `do_test`: If you finish training quickly withouth checking the outputs in the middle of training, you can set `do_test` as 0. 
- `bs_train` : batch size.
- `n_iter` : iter
- `get_h_num` : It determine the number of attributes. It should be `1` for training.
- `train_delta_block` : Train the implicit function. You can use `--train_delta_h` instead of `--train_delta_block` to optimize direction directly. (we recommend -`-train_delta_block`)
- `--save_x0`, `--use_x0_tensor` : If you want to save the results with original real images, use it.
- `n_inv_step`, `n_train_step`, `n_test_step`: # of steps during the generative pross for the inversion, training and test respectively. They are in `[0, 999]`. We usually use 40 or 1000 for `n_inv_step`, 40 or 50 for `n_train_step` and 40 or 50 or 1000 for `n_test_step` respectively.
- `clip_loss_w`, `l1_loss_w` : Weights of CLIP loss and L1 loss.

### Inference
After training finished, you can inference with various settings using `script_inference.sh`. We provide some of it.

```
python main.py  --run_test                    \
                --config $config              \
                --exp ./runs/example          \
                --edit_attr $guid             \
                --do_train 1                  \
                --do_test 1                   \
                --n_train_img 100             \
                --n_test_img 32               \
                --n_iter 5                    \
                --bs_train 1                  \
                --t_0 999                     \
                --n_inv_step 50               \
                --n_train_step 50             \
                --n_test_step $test_step      \
                --get_h_num 1                 \
                --train_delta_block           \
                --add_noise_from_xt           \
                --lpips_addnoise_th 1.2       \
                --lpips_edit_th 0.33          \
                --sh_file_name $sh_file_name  \
                --save_x0                     \
                --use_x0_tensor               \
                --hs_coeff_delta_h 1.0        \

                (optional - checkpoint)
                --load_from_checkpoint "exp_name"  
                or
                --manual_checkpoint_name "full_path.pth"

                (optional - gradually editing)
                --delta_interpolation
                --max_delta 1.0
                --min_delta -1.0
                --num_delta 10

                (optinal - multi)
                --multiple_attr "exp1 exp2 exp3"
                --multiple_hs_coeff "1 0.5 1.5"
```
- `exp` : is should be matched with trained exp. If you want to use our pretrained implicit function, you have to set `--exp` as `$guid`.
- `do_train`, `do_test`: Sampling from training dataset / test dataset.
- `n_iter` : If `n_iter` is same as trained argument, it use last-iter-checkpoint.
- `n_test_step` : You can manually regulate inference step. `1000` shows best quality.
- `hs_coeff_delta_h` : You can manually regulate the degree of editing. It can be the minus number.
- `--load_from_checkpoint`, `--manual_checkpoint_name` : `load_from_checkpoint` should be the name of exp. `manual_checkpoint_name` should be the full path of checkpoint.
- `--delta_interpolation`: You can set $max, $min, $num values. The $num of results will use gradually increased dgree of editing from min to max.
- `--multiple_attr`: If you use multiple attributes, write down the name of exps (use blanks as separators). You can use `--multiple_hs_coeff` to regulate the degree of editing respectively.


## Acknowledge
Codes are based on DiffusionCLIP.
