import torch
from torch.nn.utils.rnn import pad_sequence

def FeaturePreprocess(
    pri_feat_esmc, 
    sec_feat_dssp, 
    sec_feat_b, 
    sec_feat_sf, 
    mode: str = "train",   # Added mode parameter
    dataset = None         # Dataset is optional, required only in training mode
):
    """
    Preprocess features and return data with or without labels depending on mode.
    
    Args:
        mode (str): Mode parameter, either "train" or "predict".
        dataset (DataFrame, optional): Required only in training mode, must contain labels.
        Other parameters are consistent with the original function.
    """
    # Validate parameters
    if mode not in ["train", "predict"]:
        raise ValueError("mode must be 'train' or 'predict'")
    
    if mode == "train" and dataset is None:
        raise ValueError("dataset must be provided in training mode")

    # Common preprocessing pipeline
    seq_lengths = [seq.shape[1] for seq in pri_feat_esmc]
    max_seq_len = max(seq_lengths)
    mask = torch.zeros(len(seq_lengths), max_seq_len, dtype=torch.bool)
    for i, l in enumerate(seq_lengths):
        mask[i, :l] = True

    # Process primary features
    pri_feat_esmc = [esm.squeeze(0).cpu() for esm in pri_feat_esmc] 
    pri_feat_esmc = pad_sequence(pri_feat_esmc, batch_first=True, padding_value=0)

    # Process secondary features
    sec_feat_dssp = [torch.tensor(arr, dtype=torch.float32) for arr in sec_feat_dssp]
    sec_feat_dssp = pad_sequence(sec_feat_dssp, batch_first=True, padding_value=0)
    
    sec_feat_b = [torch.tensor(arr, dtype=torch.float32).unsqueeze(-1) for arr in sec_feat_b]
    sec_feat_b = pad_sequence(sec_feat_b, batch_first=True, padding_value=0)
    
    sec_feat_sf = [torch.tensor(arr, dtype=torch.float32).unsqueeze(-1) for arr in sec_feat_sf]
    sec_feat_sf = pad_sequence(sec_feat_sf, batch_first=True, padding_value=0)

    # Build result dictionary
    result = {
        "pri_feat_esmc": pri_feat_esmc,
        "sec_feat_dssp": sec_feat_dssp,
        "sec_feat_b": sec_feat_b,
        "sec_feat_sf": sec_feat_sf,
        "mask": mask
    }

    # Add labels in training mode
    if mode == "train":
        labels = torch.tensor(
            dataset['labels'].map({True: 1, False: 0}).values, 
            dtype=torch.long
        )
        result["labels"] = labels

    return result
