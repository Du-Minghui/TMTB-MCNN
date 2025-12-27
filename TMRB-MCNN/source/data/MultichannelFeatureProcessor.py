import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from typing import List, Optional, Dict, Union
import pandas as pd
import numpy as np

def FeaturePreprocess(
    pri_feat_esmc: List[Tensor],
    sec_feat_dssp: List[Union[np.ndarray, List[float], Tensor]],
    sec_feat_b: List[Union[np.ndarray, List[float], Tensor]],
    sec_feat_sf: List[Union[np.ndarray, List[float], Tensor]],
    mode: str = "train",   # "train" or "predict"
    dataset: Optional[pd.DataFrame] = None
) -> Dict[str, Tensor]:
    """
    Preprocess features and return a dictionary containing tensors.
    
    Args:
        pri_feat_esmc (List[Tensor]): List of primary feature tensors (ESM embeddings).
        sec_feat_dssp (List[np.ndarray | List[float] | Tensor]): Secondary DSSP features.
        sec_feat_b (List[np.ndarray | List[float] | Tensor]): Secondary B-factor features.
        sec_feat_sf (List[np.ndarray | List[float] | Tensor]): Secondary surface features.
        mode (str): Either "train" or "predict".
        dataset (pd.DataFrame, optional): Required in training mode, must contain a 'labels' column.
    
    Returns:
        Dict[str, Tensor]: Dictionary with keys:
            - "pri_feat_esmc": padded primary features
            - "sec_feat_dssp": padded DSSP features
            - "sec_feat_b": padded B-factor features
            - "sec_feat_sf": padded surface features
            - "mask": boolean mask for valid sequence positions
            - "labels": (only in training mode) tensor of labels
    """
    # Validate parameters
    if mode not in ["train", "predict"]:
        raise ValueError("mode must be 'train' or 'predict'")
    
    if mode == "train" and dataset is None:
        raise ValueError("dataset must be provided in training mode")

    # Sequence length mask
    seq_lengths = [seq.shape[1] for seq in pri_feat_esmc]
    max_seq_len = max(seq_lengths)
    mask = torch.zeros(len(seq_lengths), max_seq_len, dtype=torch.bool)
    for i, l in enumerate(seq_lengths):
        mask[i, :l] = True

    # Process primary features
    pri_feat_esmc = [esm.squeeze(0).cpu() for esm in pri_feat_esmc] 
    pri_feat_esmc = pad_sequence(pri_feat_esmc, batch_first=True, padding_value=0)

    # Process secondary features
    sec_feat_dssp = [torch.as_tensor(arr, dtype=torch.float32) for arr in sec_feat_dssp]
    sec_feat_dssp = pad_sequence(sec_feat_dssp, batch_first=True, padding_value=0)
    
    sec_feat_b = [torch.as_tensor(arr, dtype=torch.float32).unsqueeze(-1) for arr in sec_feat_b]
    sec_feat_b = pad_sequence(sec_feat_b, batch_first=True, padding_value=0)
    
    sec_feat_sf = [torch.as_tensor(arr, dtype=torch.float32).unsqueeze(-1) for arr in sec_feat_sf]
    sec_feat_sf = pad_sequence(sec_feat_sf, batch_first=True, padding_value=0)

    # Build result dictionary
    result: Dict[str, Tensor] = {
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
