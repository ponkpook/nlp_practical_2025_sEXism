import time
import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, hamming_loss


def train_epoch(model, loader, optimizer, loss_fn, device, task_type, clip_grad_norm=None):
    model.train()
    total_loss = 0
    # Use lists to store tensors from each batch
    all_preds_list, all_labels_list = [], []
    start_time = time.time()

    for iter, batch in enumerate(loader):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        #print("input_ids", batch["input_ids"].shape)
        #print("labels", batch["labels"].shape, batch["labels"].dtype)
        #print("unique labels", batch["labels"].unique())

        optimizer.zero_grad()
        logits = model(input_ids)
        #print("logits shape:", logits.shape)
        #print("labels shape:", labels.shape)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

        # Get predictions and append tensors to lists
        preds = predict_from_logits(logits, task_type)
        all_preds_list.append(preds.cpu())
        all_labels_list.append(labels.cpu())

    # Concatenate all tensors at the end of the epoch
    all_preds = torch.cat(all_preds_list, dim=0).numpy()
    all_labels = torch.cat(all_labels_list, dim=0).numpy()

    metrics = compute_metrics(all_preds, all_labels, task_type)
    metrics["loss"] = total_loss / len(loader)
    metrics["time"] = time.time() - start_time
    return metrics

@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device, task_type):
    model.eval()
    total_loss = 0
    # Use lists to store tensors from each batch
    all_preds_list, all_labels_list = [], []
    start_time = time.time()

    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        logits = model(input_ids)
        loss = loss_fn(logits, labels)
        total_loss += loss.item()

        # Get predictions and append tensors to lists
        preds = predict_from_logits(logits, task_type)
        all_preds_list.append(preds.cpu())
        all_labels_list.append(labels.cpu())

    # Concatenate all tensors at the end of the epoch
    all_preds = torch.cat(all_preds_list, dim=0).numpy()
    all_labels = torch.cat(all_labels_list, dim=0).numpy()

    metrics = compute_metrics(all_preds, all_labels, task_type)
    metrics["loss"] = total_loss / len(loader)
    metrics["time"] = time.time() - start_time
    return metrics

def predict_from_logits(logits, task_type):
    """Makes predictions based on an explicit task type."""
    #if task_type == 'binary':
    #    return torch.argmax(logits, dim=1)
    if task_type == 'binary':
        probs = torch.softmax(logits, dim=1)  # shape: [B, 2]
        class1_probs = probs[:, 1]            # probability of class 1
        return (class1_probs > 0.6).long() 
    elif task_type == 'multiclass':
        return torch.argmax(logits, dim=1)
    elif task_type == 'multilabel':
        return (torch.sigmoid(logits) > 0.5).int()
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

def compute_metrics(preds, labels, task_type):
    preds = np.array(preds)
    labels = np.array(labels)
    
    try:
        if task_type == 'multilabel':
            # Micro treats all labels equally (good for imbalance)
            micro_f1 = f1_score(labels, preds, average='micro', zero_division=0)
            macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
            weighted_f1 = f1_score(labels, preds, average='weighted', zero_division=0)
            per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)

            # Subset accuracy = all labels correct
            subset_accuracy = accuracy_score(labels, preds)

            # Hamming loss = how many bits are wrong
            hamming = hamming_loss(labels, preds)

            return {
                "accuracy": float(subset_accuracy),
                "hamming_loss": float(hamming),
                "micro_f1": float(micro_f1),
                "macro_f1": float(macro_f1),
                "weighted_f1": float(weighted_f1),
                "per_class_f1": per_class_f1.tolist()
            }
            
        else:
            if task_type == 'multiclass':
                micro_f1 = f1_score(labels, preds, average='micro', zero_division=0)
                weighted_f1 = f1_score(labels, preds, average='weighted', zero_division=0)      
            
            accuracy = accuracy_score(labels, preds)
            macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
            unique_classes = np.unique(labels)
            per_class_f1 = f1_score(labels, preds, average=None, zero_division=0, labels=unique_classes)
        
        return {
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "per_class_f1": per_class_f1.tolist() if isinstance(per_class_f1, np.ndarray) else per_class_f1,
            #"micro_f1": float(micro_f1),
            #"weighted_f1": float(weighted_f1),
        }
        
    except Exception as e:
        print(f"Could not compute metrics: {e}")
        return {"accuracy": None, "macro_f1": None, "per_class_f1": None}

def train_model(model, train_loader, val_loader, optimizer, scheduler,
                 loss_fn, device, task_type, epochs=10, log_fn=print,
                   clip_grad_norm=None, use_wandb=False, run_name=None):
    model.to(device)
    best_val = None
    epoch_times = []

    if use_wandb:
        import wandb
        wandb.init(project="my-nlp", name=run_name)
        wandb.watch(model)

    for epoch in range(epochs):
        t0 = time.time()
        # Pass the task_type down to the epoch functions
        train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, device, task_type, clip_grad_norm)
        val_metrics = eval_epoch(model, val_loader, loss_fn, device, task_type)
        #scheduler.step()

        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)
        
        # Your logging code here (it's good as is)
        log_fn(f"[Epoch {epoch+1}/{epochs}]")
        log_fn(f"Train Loss: {train_metrics['loss']:.4f}")
        if train_metrics['accuracy'] is not None:
            log_fn(f"  Train Accuracy: {100 * train_metrics['accuracy']:.2f}%")
        if train_metrics['macro_f1'] is not None:
            log_fn(f"  Train F1 Score (macro): {train_metrics['macro_f1']:.4f}")
        #if train_metrics['micro_f1'] is not None:
        #    log_fn(f"  Train F1 Score (micro): {train_metrics['micro_f1']:.4f}")
        #if train_metrics['weighted_f1'] is not None:
        #    log_fn(f"  Train F1 Score (weighted): {train_metrics['weighted_f1']:.4f}")
        #if train_metrics['hamming_loss'] is not None:
        #    log_fn(f"  Train Hamming Loss: {train_metrics['hamming_loss']:.4f}")
        if train_metrics['per_class_f1'] is not None:
            log_fn(f"  Per-Class F1 Scores: " + ', '.join(
                f"class {i} = {f1:.4f}" for i, f1 in enumerate(train_metrics['per_class_f1'])))
        log_fn()

        log_fn(f"Val Loss: {val_metrics['loss']:.4f}")
        if val_metrics['accuracy'] is not None:
            log_fn(f"  Val Accuracy: {100 * val_metrics['accuracy']:.2f}%")
        if val_metrics['macro_f1'] is not None:
            log_fn(f"  Val F1 Score (macro): {val_metrics['macro_f1']:.4f}")
        #if val_metrics['micro_f1'] is not None:
        #    log_fn(f"  Train F1 Score (micro): {val_metrics['micro_f1']:.4f}")
        #if val_metrics['weighted_f1'] is not None:
        #    log_fn(f"  Train F1 Score (weighted): {val_metrics['weighted_f1']:.4f}")
        #if val_metrics['hamming_loss'] is not None:
        #    log_fn(f"  Train Hamming Loss: {val_metrics['hamming_loss']:.4f}")
        if val_metrics['per_class_f1'] is not None:
            log_fn(f"  Per-Class F1 Scores: " + ', '.join(
                f"class {i} = {f1:.4f}" for i, f1 in enumerate(val_metrics['per_class_f1'])))
        log_fn("-" * 50)

        if use_wandb and train_metrics['macro_f1'] is not None:
            wandb.log({
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "train_f1": train_metrics["macro_f1"],
                "val_f1": val_metrics["macro_f1"],
                "epoch_time": epoch_time,
            }, step=epoch)
        
        if best_val is None or (val_metrics["macro_f1"] is not None and val_metrics["macro_f1"] > best_val["macro_f1"]):
            best_val = val_metrics

    if best_val and best_val['macro_f1'] is not None:
        log_fn(f"Best val F1: {best_val['macro_f1']:.4f}")
    log_fn(f"Avg epoch time: {np.mean(epoch_times):.2f}s")

    if use_wandb:
        wandb.finish()