import os
import datetime
import torch
import torch.distributed as dist

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.dist_url = 'env://'
    elif args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.world_size = int(os.environ.get('SLURM_NTASKS', 1))
        args.gpu = args.rank % torch.cuda.device_count()
        master_addr = os.environ.get('MASTER_ADDR', '127.0.0.1')
        master_port = os.environ.get('MASTER_PORT', '29500')
        args.dist_url = f"tcp://{master_addr}:{master_port}"
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print(f'| distributed init (rank {args.rank}): {args.dist_url}, gpu {args.gpu}', flush=True)

    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
        timeout=datetime.timedelta(seconds=5400),
    )
    torch.distributed.barrier(device_ids=[args.gpu])

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

@torch.no_grad()
def concat_all_gather(tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if not is_dist_avail_and_initialized():
        return tensor
    tensors_gather = [torch.empty_like(tensor) for _ in range(get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    return torch.cat(tensors_gather, dim=0)

def concat_all_gather_with_grad(tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs all_gather operation on the provided tensors, supporting gradient backpropagation.
    """
    if not is_dist_avail_and_initialized():
        return tensor
    
    world_size = get_world_size()
    if world_size == 1:
        return tensor

    rank = get_rank()
    
    local_size = torch.tensor([tensor.size(0)], device=tensor.device)
    size_list = [torch.tensor([0], device=tensor.device) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    sizes = torch.cat(size_list).cpu().numpy().tolist()
    max_size = max(sizes)

    if sizes[rank] < max_size:
        padding = torch.zeros((max_size - sizes[rank],) + tensor.shape[1:], 
                              device=tensor.device, dtype=tensor.dtype)
        padded_local_tensor = torch.cat([tensor, padding], dim=0)
    else:
        padded_local_tensor = tensor

    gathered_tensors = [torch.empty_like(padded_local_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, padded_local_tensor.contiguous())

    gathered_tensors[rank] = padded_local_tensor

    output_tensors = []
    for i in range(world_size):
        output_tensors.append(gathered_tensors[i][:sizes[i]])
    
    return torch.cat(output_tensors, dim=0)

def sum_all_reduce(tensor: torch.Tensor) -> torch.Tensor:
    if is_dist_avail_and_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor