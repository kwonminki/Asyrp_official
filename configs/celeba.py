import ml_collections
import torch




def get_celeba_configs():

  args = ml_collections.ConfigDict()
  args.run_each_layer_clip = True
  args.config ="celeba.yml"
  args.exp = "./run_each_layer"
  args.edit_attr = "smiling"
  args.do_train = 1             
  args.do_test = 1              
  args.n_train_img = 100         
  args.n_test_img = 20          
  args.n_iter = 4               
  args.bs_train = 4         
  args.t_0 = 999                
  args.n_inv_step = 40          
  args.n_train_step = 40         
  args.n_test_step = 40         
  args.get_h_num = 1           
  args.lr_latent_clr = 1e-1     
  args.id_loss_w = 1            
  args.clip_loss_w = 1         
  args.l1_loss_w = 3        
  args.maintain = 295       
  args.save_train_image_step = 6 
  args.interpolation_step = 8 
  args.retrain = 1          
  args.scheduler_step_size = 4 
  args.aimed_index = "8"  
    # Default
  args.seed =1234
  args.exp ='./runs/'
  args.comment =''
  args.verbose ='info'
  args.ni =1
  args.align_face =1
  args.sample_type='ddim'
  





  
  return args