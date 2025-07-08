import os
import argparse

import multiprocessing as mp

import pprint
import yaml

parser = argparse.ArgumentParser()
parser.add_argument(
    "--fname", type=str,
    help="name of config file to load",
    default="configs.yaml"
)
parser.add_argument(
    "--task", type=str, choices=["train_dist", "train", "test"],
)
parser.add_argument(
    "--devices", type=str, nargs="+", default=["cuda:0"],
)
parser.add_argument(
    "--port", type=int, default=29500,
)

# For test
parser.add_argument("--save_dir", type=str, default=None,)
parser.add_argument("--eval_strategy", type=str, default="inversion",)
parser.add_argument("--eval_step", type=int, default=3)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument('--noise_step', type=int, default=8, help='Number of noise steps for evaluation')
parser.add_argument('--use_ema_model', action='store_true', help='Use EMA model for evaluation')
parser.add_argument('--use_best_model', action='store_true', help='Use best model for evaluation')


def process_main(rank, fname, world_size, devices, task, port):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(":")[-1])
    
    import torch
    torch.cuda.set_device(0)
    import torch.distributed as dist
    from src.utils import init_distributed
    
    import logging
    logging.basicConfig()
    logger = logging.getLogger()
    
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)
    
    logging.info(f"called-params {fname}")  
    
    # load params
    params = None
    with open(fname, "r") as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logging.info("loaded params...")
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)
    
    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size), port=port)
    logger.info(f"Running... (rank: {rank}/{world_size})")
    if task == "train_dist":
        from src.train_distributed import main as train_dist_main
        train_dist_main(params)
    elif task == "train":
        from src.train import main as train_main
        train_main(params)
    elif task == "test":
        from src.evaluate import main as test_main
        test_main(params, args)
    else:
        raise ValueError(f"Task {task} should be specified")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    args = parser.parse_args()

    if "test" in args.task:
        args.fname = os.path.join(args.save_dir, "config.yaml")
    
    if "dist" not in args.task:
        args.devices = ["cuda:0"]
        process_main(0, args.fname, 1, args.devices, args.task, args.port)
        exit(0)
    
    num_gpus = len(args.devices)
    mp.set_start_method("spawn", True)
    
    processes = []
    for rank in range(num_gpus):
        p = mp.Process(
            target=process_main,
            args=(rank, args.fname, num_gpus, args.devices, args.task, args.port)
        )
        p.start()
        processes.append(p)
    
    # wait for all processes to finish
    for p in processes:
        p.join()