import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Type

def model_predict(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    feature_masking: bool = False
) -> Dict[str, list]:
    """
    Run model prediction over a dataset.

    Args:
        model (nn.Module): Trained PyTorch model.
        data_loader (DataLoader): DataLoader providing batches of features.
        device (str): Device string, e.g. 'cuda' or 'cpu'.
        feature_masking (bool): If True, mask out secondary features.

    Returns:
        Dict[str, list]: Dictionary containing:
            - 'probs': list of probability arrays for each sample
            - 'pred_labels': list of predicted class labels
    """
    model.eval()
    predictions = {'probs': [], 'pred_labels': []}
    
    with torch.no_grad():
        for batch in data_loader:
            try:
                # Assume the dataloader returns in order: (esmc, dssp, b, sf, mask)
                esmc, dssp, b, sf, mask = batch[:5]  # does not include labels
                
                # Preprocess B-factors
                b = -torch.log10(b + 1.0)
                
                # Move data to device
                inputs = (
                    esmc.to(device),
                    dssp.to(device) if not feature_masking else None,
                    b.to(device) if not feature_masking else None,
                    sf.to(device) if not feature_masking else None,
                    mask.to(device)
                )
                
                # Model inference
                if feature_masking:
                    outputs = model(esmc=inputs[0], dssp=None, b=None, sf=None, mask=inputs[4])
                else:
                    outputs = model(esmc=inputs[0], dssp=inputs[1], b=inputs[2], sf=inputs[3], mask=inputs[4])
                
                # Collect predictions
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                pred_labels = np.argmax(probs, axis=1)
                
                predictions['probs'].extend(probs)
                predictions['pred_labels'].extend(pred_labels)
                
            except Exception as e:
                print(f"Skipping invalid batch during prediction: {str(e)}")
                continue
                
    return predictions


def predict(
    dataset: Any,
    model_class: Type[nn.Module],
    batch_size: int = 64,
    device: str = 'cuda',
    feature_masking: bool = False,
    model_path: str = 'best_model.pth'
) -> Dict[str, np.ndarray]:
    """
    Full prediction pipeline.

    Args:
        dataset (Dataset): PyTorch dataset containing features.
        model_class (Type[nn.Module]): Model class to instantiate.
        batch_size (int): Batch size for prediction.
        device (str): Device string, e.g. 'cuda' or 'cpu'.
        feature_masking (bool): If True, mask out secondary features.
        model_path (str): Path to the saved model weights.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing:
            - 'probabilities': numpy array of shape (n_samples, n_classes)
            - 'predicted_labels': numpy array of shape (n_samples,)
    """
    # Create dataloader (no shuffle needed)
    predict_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    init_dims = 256 if feature_masking else 384
    model = model_class(init_dims).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded model weights: {model_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")
    
    # Run prediction
    predictions = model_predict(
        model=model,
        data_loader=predict_loader,
        device=device,
        feature_masking=feature_masking
    )
    
    # Convert to numpy arrays
    return {
        'probabilities': np.array(predictions['probs']),
        'predicted_labels': np.array(predictions['pred_labels'])
    }
