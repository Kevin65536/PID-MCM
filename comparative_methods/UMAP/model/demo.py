from transformers import Blip2QFormerConfig
from config import Config, get_param_sets
from UMAP_pretrain import UMAPPretrain
from UMAP_finetune import UMAPFinetune
import torch

## ===========pretrain=============

config_my = Config('config_pretrain.yaml')
config_my = get_param_sets(config_my)
config_my = config_my[0]

device = torch.device('cuda')
config = Blip2QFormerConfig(**config_my)
model = UMAPPretrain(config,
                umap_device=device, 
                seq_length=config.seq_length,
                eeg_input_dim=config.eeg_input_dim,
                eye_input_dim=config.eye_input_dim).to(device)

eeg = torch.rand(256,5,310).to(device)
eye = torch.rand(256,5,33).to(device)

loss, loss_con, loss_mat, lm_loss = model(eeg,eye)
print(loss, loss_con, loss_mat, lm_loss)



## =============finetune. EEG is missing=================

config_my = Config('config_finetune.yaml')
config_my = get_param_sets(config_my)
config_my = config_my[0]

device = torch.device('cuda')
config = Blip2QFormerConfig(**config_my)
model = UMAPFinetune(config,
            umap_device=device,
            seq_length=config.seq_length,
            eeg_input_dim=config.eeg_input_dim,
            eye_input_dim=config.eye_input_dim,
            n_class=config.n_class,
            mode='eye').to(device)

# # load pretrain weights
# checkpoint_path ='your_path/checkpoint-199.pth'
# checkpoint = torch.load(checkpoint_path, map_location='cpu') 
# state_dict = model.state_dict()  
# for k,v in state_dict.items():
#     if k in checkpoint['model']:
#         state_dict[k] = checkpoint['model'][k]
# model.load_state_dict(state_dict) 
# ## frozen parameters  
# for name, param in model.named_parameters():
#     if 'eeg' in name:
#         param.requires_grad = False

eye = torch.rand(256,5,33).to(device)
logits = model(eeg=None,eye=eye)
print(logits.shape)