import os
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.cliff_hr48.cliff import CLIFF as cliff_hr48
from models.cliff_res50.cliff import CLIFF as cliff_res50
from common import constants
from common.utils import strip_prefix_if_present
from common.mocap_dataset import MocapDataset

def train(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Create the model instance
    cliff = eval("cliff_" + args.backbone)
    model = cliff(constants.SMPL_MEAN_PARAMS).to(device)
    
    # Load the pretrained model if specified
    if args.ckpt:
        print("Load the CLIFF checkpoint from path:", args.ckpt)
        state_dict = torch.load(args.ckpt)['model']
        state_dict = strip_prefix_if_present(state_dict, prefix="module.")
        model.load_state_dict(state_dict, strict=True)
    
    # Set up the dataset and data loader
    train_dataset = MocapDataset(args.train_data)
    val_dataset = MocapDataset(args.val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Set up the loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{args.epochs}"):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{args.epochs}], Train Loss: {avg_train_loss:.4f}")
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{args.epochs}], Validation Loss: {avg_val_loss:.4f}")
    
    # Save the trained model
    torch.save(model.state_dict(), args.save_model)
    print(f"Model saved to {args.save_model}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', required=True, help='Path to the training data')
    parser.add_argument('--val_data', required=True, help='Path to the validation data')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--ckpt', help='Path to the pretrained checkpoint')
    parser.add_argument('--backbone', default='hr48', choices=['res50', 'hr48'], help='Backbone architecture')
    parser.add_argument('--save_model', default='cliff_model.pth', help='Path to save the trained model')
    args = parser.parse_args()
    train(args)
