from torch.utils import data

# =======================================================
# Main function to get dataloader from dataset
# =======================================================


def _acquire_data_loader(
    dataset,
    train,
    batch_size,
    num_workers,
    rank=0,
    world_size=1,
    drop_last=False,
    tpu=False,
):
    # Adapted from: https://github.com/pytorch/xla/blob/56138cf7b29dc20ed9b0ca5934b91d1cf9a72b70/test/test_train_mp_imagenet.py#L149
    assert isinstance(dataset, data.Dataset)
    sampler = data.distributed.DistributedSampler(
        dataset=dataset, num_replicas=world_size, rank=rank, shuffle=train
    )
    if tpu:
        loader_kwargs = {"num_workers": num_workers}
    else:
        loader_kwargs = {"num_workers": num_workers, "pin_memory": True}

    # shuffle always set to False since we passed it to the sampler already
    loader_kwargs["shuffle"] = False
    # drop last here is to ensure the number of examples is the same for each minibatch
    # (e.g. dataset size evenly divides batch size)
    # Drop last in distributed sampler is the decision whether or not to drop the last set
    # of examples to ensure the dataset size evenly divides the number of replica).
    # In this case, drop_last=False for the DistributedSampler adds examples to ensure
    # the dataset size evenly divides the number of replicas.
    # Therefore, we keep drop_last=False in DistributedSampler to sample as many distinct examples as possible.
    loader_kwargs["drop_last"] = drop_last
    loader = data.DataLoader(
        dataset=dataset, batch_size=batch_size, sampler=sampler, **loader_kwargs
    )
    return loader

# =======================================================
# Wrapper for getting dataloaders
# =======================================================
def wrap_dataloaders(dataloader_func, params, my_transforms, device, rank=0, world_size=1):
    tpu = (device.type == "xla")

    assert params["train_batch_size"] % world_size == 0
    if "val_batch_size" in params.keys():
        assert params["val_batch_size"] % world_size == 0
    # for TPU and GPU we do multiprocessing, so batch size is per GPU/TPU core
    params["train_batch_size"] = params["train_batch_size"] // world_size
    if "val_batch_size" in params.keys():
        params["val_batch_size"] = params["val_batch_size"] // world_size

    train_loader, val_loader = dataloader_func(
        params=params,
        my_transforms=my_transforms,
        rank=rank,
        world_size=world_size,
        tpu=tpu,
    )

    if tpu:
        import torch_xla.distributed.parallel_loader as pl

        train_loader = pl.MpDeviceLoader(loader=train_loader, device=device)
        if val_loader is not None:
            val_loader = pl.MpDeviceLoader(loader=val_loader, device=device)

    return train_loader, val_loader
