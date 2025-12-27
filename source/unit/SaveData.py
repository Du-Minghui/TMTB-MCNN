import pandas as pd
import numpy as np

def save_metrics(results):

    all_metrics = []
    
    for fold_idx, fold_data in enumerate(results):
        for epoch_data in fold_data:
            row = {
                'fold': fold_idx + 1,  
                'epoch': epoch_data['epoch'],
                'train_loss': epoch_data['train_loss'],
                'val_loss': epoch_data['val_loss'],
                'train_accuracy': epoch_data['train_metrics']['accuracy'],
                'train_precision': epoch_data['train_metrics']['precision'],
                'train_recall': epoch_data['train_metrics']['recall'],
                'train_auc': epoch_data['train_metrics']['auc'],
                'val_accuracy': epoch_data['val_metrics']['accuracy'],
                'val_precision': epoch_data['val_metrics']['precision'],
                'val_recall': epoch_data['val_metrics']['recall'],
                'val_auc': epoch_data['val_metrics']['auc']
            }
            all_metrics.append(row)
    
    return pd.DataFrame(all_metrics)


def save_predictions(results, n_classes=2):

    all_dfs = []

    for fold_idx, fold_data in enumerate(results):
        for epoch_data in fold_data:
            epoch = epoch_data['epoch']
            
            # Process each dataset type
            for dataset_type in ['train', 'val']:
                metrics_key = f"{dataset_type}_metrics"
                
                try:
                    labels = epoch_data[metrics_key]['labels']
                    probs = epoch_data[metrics_key]['probs']
                    preds = epoch_data[metrics_key]['preds']
                except KeyError as e:
                    print(f"Skipping missing data: {e}")
                    continue
                
                # Dynamically generate probability column names
                prob_cols = {f'prob_{i}': probs[:, i] for i in range(n_classes)}
                
                df = pd.DataFrame({
                    'fold': fold_idx + 1,
                    'epoch': epoch,
                    'dataset': dataset_type,
                    'label': labels,
                    'pred': preds
                })
                
                # Add probability columns
                df = df.assign(**prob_cols)
                all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
