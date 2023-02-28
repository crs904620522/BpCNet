import torch
import torch.optim as optim
import numpy as np
import os
import argparse
import time
import matplotlib;
matplotlib.use('Agg')
from lf2disp import config, data
from lf2disp.checkpoints import CheckpointIO
if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(
        description='Train a light field disparity estimation model.'
    )
    parser.add_argument('--config', type=str, default='./configs/EPINET_UrbanLF.yaml')
    parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
    parser.add_argument('--exit-after', type=int, default=-1,
                        help='Checkpoint and exit after specified number of seconds'
                             'with exit code 2.')

    args = parser.parse_args()
    cfg = config.load_config(args.config, 'configs/default.yaml')
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg['GPU_ID']
    is_cuda = (torch.cuda.is_available() and not args.no_cuda)
    device = torch.device("cuda" if is_cuda else "cpu")
    # Set t0
    t0 = time.time()

    # Shorthands
    out_dir = cfg['training']['out_dir']
    batch_size = cfg['training']['batch_size']
    exit_after = args.exit_after

    # Set Training metric
    model_selection_metric = cfg['training']['model_selection_metric']
    if cfg['training']['model_selection_mode'] == 'maximize':
        model_selection_sign = 1
    elif cfg['training']['model_selection_mode'] == 'minimize':
        model_selection_sign = -1
    else:
        raise ValueError('model_selection_mode must be '
                         'either maximize or minimize.')

    val_metric = cfg['training']['val_metric']

    # Output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Dataset
    train_dataset = config.get_dataset('train', cfg)
    test_dataset = config.get_dataset('test', cfg)
    vis_dataset = config.get_dataset('vis', cfg)

    # Model
    model = config.get_model(cfg, device=device, dataset=train_dataset)

    # Intialize training
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # Set trainer
    trainer = config.get_trainer(model, optimizer, cfg, device=device)

    checkpoint_io = CheckpointIO(out_dir, model=model, optimizer=optimizer)
    try:
        load_dict = checkpoint_io.load('model.pt')
    except FileExistsError:
        load_dict = dict()
    epoch_it = load_dict.get('epoch_it', -1)
    it = load_dict.get('it', -1)
    metric_val_best = load_dict.get(
        'loss_val_best', -model_selection_sign * np.inf)

    if metric_val_best == np.inf or metric_val_best == -np.inf:
        metric_val_best = -model_selection_sign * np.inf

    print('Current best validation metric (%s): %.8f'
          % (model_selection_metric, metric_val_best))

    # Shorthands
    print_every = cfg['training']['print_every']
    checkpoint_every = cfg['training']['checkpoint_every']
    validate_every = cfg['training']['validate_every']
    visualize_every = cfg['training']['visualize_every']
    backup_every = cfg['training']['backup_every']

    # Print model
    nparameters = sum(p.numel() for p in model.parameters())
    print('Total number of parameters: %d' % nparameters)

    # dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=4, shuffle=True,
        collate_fn=data.collate_remove_none,
        worker_init_fn=data.worker_init_fn)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, num_workers=0, shuffle=False,
        collate_fn=data.collate_remove_none,
        worker_init_fn=data.worker_init_fn)

    # For visualizations
    vis_loader = torch.utils.data.DataLoader(
        vis_dataset, batch_size=1, shuffle=False,
        collate_fn=data.collate_remove_none,
        worker_init_fn=data.worker_init_fn)

    start = time.time()
    while True:
        epoch_it += 1

        for batch in train_loader:
            it += 1
            # torch.backends.cudnn.enabled = False
            torch.backends.cudnn.enabled = True
            loss = trainer.train_step(batch, iter=it)

            # Print output
            if print_every > 0 and (it % print_every) == 0:
                print('[Epoch %02d] it=%03d, loss=%.4f, time=%.4f'
                      % (epoch_it, it, loss,time.time()-start))

            # Visualize output
            if visualize_every > 0 and (it % visualize_every) == 0:
                # torch.backends.cudnn.enabled = False
                print('Visualizing')
                for i, batch in enumerate(vis_loader):
                    torch.cuda.empty_cache()
                    trainer.visualize(batch, id=i)

            # Save checkpoint
            if checkpoint_every > 0 and (it % checkpoint_every) == 0:
                print('Saving checkpoint')
                checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                                   loss_val_best=metric_val_best)

            # Backup if necessary
            if backup_every > 0 and (it % backup_every) == 0:
                print('Backup checkpoint')
                checkpoint_io.save('model_%d.pt' % it, epoch_it=epoch_it, it=it,
                                   loss_val_best=metric_val_best)

            # Run validation
            if validate_every > 0 and (it % validate_every) == 0:
                # torch.backends.cudnn.enabled = False
                try:
                    eval_dict = trainer.evaluate(test_loader)
                    metric_val = eval_dict[val_metric]
                    print('Validation metric (%s): %.4f'
                          % (val_metric, metric_val))
                    print("All mean", eval_dict)

                    if model_selection_sign * (metric_val - metric_val_best) > 0:
                        metric_val_best = metric_val
                        print('New best model (loss %.4f)' % metric_val_best)
                        checkpoint_io.save('model_best.pt', epoch_it=epoch_it, it=it,
                                           loss_val_best=metric_val_best)
                except Exception as e:
                    print(e)
                    continue

            # Exit if necessary
            if exit_after > 0 and (time.time() - t0) >= exit_after:
                print('Time limit reached. Exiting.')
                checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                                   loss_val_best=metric_val_best)
                exit(3)
