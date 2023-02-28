import torch
import os
import argparse
from tqdm import tqdm
from collections import defaultdict
from lf2disp import config
from lf2disp.checkpoints import CheckpointIO
import numpy as np

parser = argparse.ArgumentParser(
    description='Test a lightfield disparity estimation model.'
)
parser.add_argument('--config', type=str, default='./configs/pretrained/BpCNet_pretrained.yaml')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--exit-after', type=int, default=-1,
                    help='Checkpoint and exit after specified number of seconds'
                         'with exit code 2.')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
os.environ["CUDA_VISIBLE_DEVICES"] = cfg['GPU_ID']
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

out_dir = cfg['training']['out_dir']
# Dataset
dataset = config.get_dataset('generate', cfg) # test:8 generate:12  vis:all
# Model
model = config.get_model(cfg, device=device, dataset=dataset)
# Load checkpoint
checkpoint_io = CheckpointIO(out_dir, model=model)
checkpoint_io.load(cfg['generation']['model_file'])

# Generator
generator = config.get_generator(model, cfg, device=device)

generate_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, num_workers=0, shuffle=False)

# Statistics
time_dicts = []

# Generate
model.eval()

# Count how many models already created
model_counter = defaultdict(int)
eval_list = defaultdict(list)

for it, data in enumerate(tqdm(generate_loader)):
    out = generator.generate_depth(data, id=it)
    for k, v in out.items():
        eval_list[k].append(v)
    torch.cuda.empty_cache()
eval_dict = {k: np.mean(v) for k, v in eval_list.items()}

print("mean", eval_dict)
print("Finish")
