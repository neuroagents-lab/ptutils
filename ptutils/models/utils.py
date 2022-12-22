import torch
from collections import OrderedDict

def load_model(
    model,
    trained=False,
    model_path=None,
    state_dict_key="state_dict",
    **kwargs,
):
    # Load weights if params file is given.
    if model_path is not None and trained:
        try:
            params = torch.load(model_path, map_location="cpu")
        except:
            raise ValueError(f"Could not open file: {model_path}")

        assert (
            state_dict_key in params.keys()
        ), f"{state_dict_key} not in params dictionary."
        sd = params[state_dict_key]
        # adapted from: https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/13
        new_sd = OrderedDict()
        for k, v in sd.items():
            if k.startswith("module."):
                name = k[7:]  # remove 'module.' of dataparallel/DDP
            else:
                name = k
            new_sd[name] = v
        model.load_state_dict(new_sd)
        print(f"Loaded parameters from {model_path}")

    # Set model to eval mode
    model.eval()

    return model