import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def model_predict(model, data_loader, device, feature_masking=False):
    """Run model prediction"""
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
    dataset,
    model_class,
    batch_size=64,
    device='cuda',
    feature_masking=False,
    model_path='best_model.pth'
):
    """
    Full prediction pipeline.
    
    Returns:
        dict: Contains two keys
            - 'probabilities': probability matrix of shape (n_samples, n_classes)
            - 'predicted_labels': predicted label array of shape (n_samples,)
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
