from model import Bert
from data_process import AmazonReviewsDatasetSplit
import torch
import logging
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

def train(model_path, csv_file, max_len, batch_size, num_epochs, learning_rate, device, save_dir='./trained_model', log_dir='./logs'):
    # Set up logger
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  
            logging.FileHandler(os.path.join(log_dir, 'training.log'))  
        ]
    )
    logger = logging.getLogger()

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Set up TensorBoard
    writer = SummaryWriter(log_dir=log_dir)

    # Load pre-trained model and tokenizer
    model = Bert(model_path)
    tokenizer = model.get_tokenizer()
    
    # Split data into train, val and test sets
    data = AmazonReviewsDatasetSplit(csv_file, tokenizer, max_len)
    train_dataset, val_dataset, test_dataset = data.get_data()
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    
    # Initialize AMP scaler
    scaler = torch.cuda.amp.GradScaler()

    # Train model
    model.to(device)
    step = 0  # Global step counter
    best_val_loss = float('inf')  # Initialize best validation loss
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        # Progress bar for training
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")
        for i, data in enumerate(progress_bar):
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            labels = data['label'].to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision forward and backward pass
            with torch.cuda.amp.autocast():
                outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                loss = criterion(outputs[1], labels)  # Adjusted for the output format of the model
            
            # Scale loss for AMP
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            step += 1
            

            
            # Update progress bar and logger
            progress_bar.set_postfix(loss=loss.item())
            
            # Log loss to TensorBoard
            if step % 10 == 0:
                writer.add_scalar('Loss/Train', loss.item(), step)
                
            if step % 100 == 0:
                logger.info(f'Step {step}, Loss: {loss.item()}')
                
            # Save and check validation loss every 1000 steps
            if step % 1000 == 0:
                model.eval()
                val_loss = 0
                all_preds = []
                all_labels = []
                with torch.no_grad():
                    for val_data in val_loader:
                        input_ids = val_data['input_ids'].to(device)
                        attention_mask = val_data['attention_mask'].to(device)
                        token_type_ids = val_data['token_type_ids'].to(device)
                        labels = val_data['label'].to(device)
                        
                        with torch.cuda.amp.autocast():
                            val_outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                            val_loss += criterion(val_outputs[1], labels).item()
                            
                            # Collect predictions and labels
                            predictions = torch.argmax(val_outputs[1], dim=1)
                            all_preds.extend(predictions.cpu().numpy())
                            all_labels.extend(labels.cpu().numpy())
                
                # Calculate metrics
                val_loss /= len(val_loader)
                val_precision = precision_score(all_labels, all_preds, average='weighted')
                val_recall = recall_score(all_labels, all_preds, average='weighted')
                val_f1 = f1_score(all_labels, all_preds, average='weighted')
                
                # Classification report
                class_report = classification_report(
                    all_labels, 
                    all_preds, 
                    target_names=[f'Class {i}' for i in range(len(set(all_labels)))],
                    digits=4
                )
                
                # Log metrics to TensorBoard
                writer.add_scalar('Loss/Validation', val_loss, step)
                writer.add_scalar('Metrics/Precision', val_precision, step)
                writer.add_scalar('Metrics/Recall', val_recall, step)
                writer.add_scalar('Metrics/F1', val_f1, step)
                logger.info(
                    f'Step {step}, Validation Loss: {val_loss}, Precision: {val_precision}, Recall: {val_recall}, F1: {val_f1}'
                )
                logger.info(f'\nClassification Report at step {step}:\n{class_report}')
                
                # Save best model based on validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_path = os.path.join(save_dir,  f'model_step_{step}.pth')
                    torch.save(model.state_dict(), save_path)
                    logger.info(
                        f'Best model saved at step {step} with validation loss: {val_loss}, F1: {val_f1}'
                    )
                
                model.train()  # Return to training mode
        
        # Step the learning rate scheduler
        scheduler.step()
        logger.info(f'Epoch {epoch + 1}, Learning Rate: {scheduler.get_last_lr()[0]}')
    
    # Close TensorBoard writer
    writer.close()
            