import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import KFold

import source.data.MultichannelFeatureProcessor as mcfeat

def model_train(model, train_loader, optimizer, criterion, device, feature_masking=False, grad_clip=None):
    """Run one training epoch"""
    model.train()
    epoch_data = {'losses': [], 'probs': [], 'labels': []}
    
    for batch in train_loader:
        try:
            # Unpack batch (adjust according to actual data structure)
            esmc, dssp, b, sf, mask, labels = batch
            # b = -torch.log(b + 1.0)
            # b = torch.log(b + 1.0)
            b = -torch.log10(b + 1.0)
            # b = torch.log10(b + 1.0)
            
            # Move data to device
            esmc, dssp, b, sf, mask, labels = (
                esmc.to(device), 
                dssp.to(device),  
                b.to(device), 
                sf.to(device), 
                mask.to(device), 
                labels.to(device)
            )
            
            # Forward pass
            if feature_masking:
                outputs = model(esmc=esmc, dssp=None, b=None, sf=None, mask=mask)
            else:
                outputs = model(esmc=esmc, dssp=dssp, b=b, sf=sf, mask=mask)
            
            loss = criterion(outputs, labels)
            
            # Record data
            epoch_data['losses'].append(loss.item())
            epoch_data['probs'].extend(torch.softmax(outputs.detach(), dim=1).cpu().numpy())
            epoch_data['labels'].extend(labels.cpu().numpy())
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
            optimizer.step()
            
        except ValueError as e:
            print(f"Skipping invalid training batch: {e}")
            continue
            
    return epoch_data


def model_validate(model, val_loader, criterion, device, feature_masking=False):
    """Run validation"""
    model.eval()
    epoch_data = {'losses': [], 'probs': [], 'labels': []}
    
    with torch.no_grad():
        for batch in val_loader:
            try:
                esmc, dssp, b, sf, mask, labels = batch
                # b = -torch.log(b + 1.0)
                # b = torch.log(b + 1.0)
                b = -torch.log10(b + 1.0)
                # b = torch.log10(b + 1.0)
                
                esmc, dssp, b, sf, mask, labels = (
                    esmc.to(device), 
                    dssp.to(device), 
                    b.to(device), 
                    sf.to(device), 
                    mask.to(device), 
                    labels.to(device)
                )
                
                if feature_masking:
                    outputs = model(esmc=esmc, dssp=None, b=None, sf=None, mask=mask)
                else:
                    outputs = model(esmc=esmc, dssp=dssp, b=b, sf=sf, mask=mask)
                    
                loss = criterion(outputs, labels)
                
                epoch_data['losses'].append(loss.item())
                epoch_data['probs'].extend(torch.softmax(outputs, dim=1).cpu().numpy())
                epoch_data['labels'].extend(labels.cpu().numpy())
                
            except ValueError as e:
                print(f"Skipping invalid validation batch: {e}")
                continue
                
    return epoch_data


def calculate_metrics(data_dict):
    """Calculate evaluation metrics"""
    labels = np.array(data_dict['labels'])
    probs = np.array(data_dict['probs'])
    preds = np.argmax(probs, axis=1)
    
    metrics = {
        'labels': labels,
        'probs': probs,
        'preds': preds,
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds),
        'recall': recall_score(labels, preds),
        'auc': roc_auc_score(labels, probs[:, 1])
    }
    return metrics


def cross_validation_train(
    dataset,
    model_class,
    n_splits,
    num_epochs=1000,
    lr=1e-4,
    weight_decay=1e-4,
    batch_size=64,
    device='cuda',
    grad_clip=None,
    patience=3,
    feature_masking=False,
    save_path=None
):
    from copy import deepcopy

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    splits = kf.split(X=np.arange(len(dataset)), y=dataset.labels.numpy())

    all_results = []
    best_global_acc = 0.0
    best_global_model_state = None
    best_global_info = {}

    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\n===== Fold {fold+1} =====")

        train_labels = dataset.labels[train_idx]
        n_pos = (train_labels == 1).sum().item()
        n_neg = (train_labels == 0).sum().item()
        class_weights = torch.tensor([1.0, n_neg / n_pos], device=device)

        init_dims = 256 if feature_masking else 384
            
        model = model_class(init_dims).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1, weight=class_weights)

        train_loader, val_loader = mcfeat.Featureloader(
            dataset, train_idx, val_idx, batch_size=batch_size
        )

        fold_results = []
        best_val_acc = 0.0
        best_val_auc = 0.0
        best_val_ppv = 0.0
        best_val_tpr = 0.0
        best_model_state = None
        best_epoch = 0
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            train_data = model_train(model, train_loader, optimizer, criterion, device, feature_masking, grad_clip)
            val_data = model_validate(model, val_loader, criterion, device, feature_masking)

            train_metrics = calculate_metrics(train_data)
            val_metrics = calculate_metrics(val_data)

            epoch_result = {
                'epoch': epoch + 1,
                'train_loss': np.mean(train_data['losses']),
                'val_loss': np.mean(val_data['losses']),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }
            fold_results.append(epoch_result)

            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"Train Loss: {epoch_result['train_loss']:.4f} | Acc: {train_metrics['accuracy']:.4f} | PPV: {train_metrics['precision']:.4f} | TPR: {train_metrics['recall']:.4f} | AUC: {train_metrics['auc']:.4f}")
            print(f"Val Loss: {epoch_result['val_loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | PPV: {val_metrics['precision']:.4f} | TPR: {val_metrics['recall']:.4f} | AUC: {val_metrics['auc']:.4f}")
        
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                best_val_auc = val_metrics['precision']
                best_val_ppv = val_metrics['recall']
                best_val_tpr = val_metrics['auc']
                best_epoch = epoch + 1
                best_model_state = deepcopy(model.state_dict())
                # torch.save(best_model_state, f"best_model_fold{fold+1}.pth")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1} due to no improvement for {patience} consecutive epochs. | best val acc = {best_val_acc}")
                break

        all_results.append(fold_results)

        if best_val_acc > best_global_acc:
            best_global_acc = best_val_acc
            best_global_model_state = deepcopy(best_model_state)
            best_global_info = {
                'fold': fold + 1,
                'epoch': best_epoch,
                'accuracy': best_val_acc,
                'precision': best_val_ppv,
                'recall': best_val_tpr,
                'auc': best_val_auc
            }
            
        print("\n==== Global Best Model ====")
        print(f"Saved global best model from Fold {best_global_info['fold']} at Epoch {best_global_info['epoch']}")
        print(f"Accuracy: {best_global_info['accuracy']:.4f} | Precision: {best_global_info['precision']:.4f} | Recall: {best_global_info['recall']:.4f} | AUC: {best_global_info['auc']:.4f}")

    # Save global best model
    if save_path is not None and best_global_model_state is not None:
        torch.save(best_global_model_state, save_path)
        print(f"\nSaved global best model to {save_path} with accuracy: {best_global_acc:.4f}")

    
    return all_results
