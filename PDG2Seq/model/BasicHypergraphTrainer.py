import torch
import math
import os
import time
import copy
import numpy as np
from lib.logger import get_logger
from lib.metrics import All_Metrics
import pandas as pd


class BasicHypergraphTrainer(object):
    def __init__(self, model, loss, optimizer, train_loader, val_loader, test_loader,
                 scaler, args, lr_scheduler=None, device='cpu', distance_matrix=None):
        super(BasicHypergraphTrainer, self).__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.args = args
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.distance_matrix = distance_matrix
        
        # Move distance matrix to device if provided
        if self.distance_matrix is not None:
            self.distance_matrix = torch.tensor(self.distance_matrix, dtype=torch.float32).to(device)
        
        self.train_per_epoch = len(train_loader)
        if val_loader != None:
            self.val_per_epoch = len(val_loader)
        
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')
        self.best_test_path = os.path.join(self.args.log_dir, 'best_test_model.pth')
        
        #log
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        self.logger.info(args)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        self.batches_seen = 0

    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        total_val_loss = 0
        epoch_time = time.time()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_dataloader):
                data = data.to(self.device)
                target = target.to(self.device)
                label = target[..., :self.args.output_dim].clone()
                
                # Extract temporal information from data if available
                if data.shape[-1] > self.args.input_dim:
                    # Time-in-day and day-in-week are in the last dimensions
                    tid_data = data[..., self.args.input_dim:self.args.input_dim+1]  # time-in-day
                    diw_data = data[..., -1:]  # day-in-week
                    
                    # Convert continuous time features to indices for embedding
                    # Assuming time_in_day is normalized [0,1], convert to [0,287] for 5-min intervals
                    tid = (tid_data.squeeze(-1) * 287).long()  # (B, T)
                    diw = diw_data.squeeze(-1).long()  # (B, T)
                else:
                    tid = None
                    diw = None
                
                # Get pure input data (remove temporal features)
                input_data = data[..., :self.args.input_dim]
                
                # Forward pass
                output = self.model(input_data, tid, diw, distance_matrix=self.distance_matrix)
                
                if self.args.real_value:
                    output = self.scaler.inverse_transform(output)
                
                loss = self.loss(output, label)
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
                    
        val_loss = total_val_loss / len(val_dataloader)
        self.logger.info('***********Val Epoch {}: average Loss: {:.6f}, time: {:.2f} s'.format(
            epoch, val_loss, time.time() - epoch_time))
        return val_loss

    def test_epoch(self, epoch, test_dataloader):
        self.model.eval()
        total_test_loss = 0
        epoch_time = time.time()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_dataloader):
                data = data.to(self.device)
                target = target.to(self.device)
                label = target[..., :self.args.output_dim].clone()
                
                # Extract temporal information
                if data.shape[-1] > self.args.input_dim:
                    tid_data = data[..., self.args.input_dim:self.args.input_dim+1]
                    diw_data = data[..., -1:]
                    tid = (tid_data.squeeze(-1) * 287).long()
                    diw = diw_data.squeeze(-1).long()
                else:
                    tid = None
                    diw = None
                
                input_data = data[..., :self.args.input_dim]
                output = self.model(input_data, tid, diw, distance_matrix=self.distance_matrix)
                
                if self.args.real_value:
                    output = self.scaler.inverse_transform(output)
                
                loss = self.loss(output, label)
                if not torch.isnan(loss):
                    total_test_loss += loss.item()
                    
        test_loss = total_test_loss / len(test_dataloader)
        self.logger.info('**********Test Epoch {}: average Loss: {:.6f}, time: {:.2f} s'.format(
            epoch, test_loss, time.time() - epoch_time))
        return test_loss

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        epoch_time = time.time()
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.batches_seen += 1
            data = data.to(self.device)
            target = target.to(self.device)
            label = target[..., :self.args.output_dim].clone()
            
            # Extract temporal information
            if data.shape[-1] > self.args.input_dim:
                tid_data = data[..., self.args.input_dim:self.args.input_dim+1]
                diw_data = data[..., -1:]
                tid = (tid_data.squeeze(-1) * 287).long()
                diw = diw_data.squeeze(-1).long()
            else:
                tid = None
                diw = None
                
            input_data = data[..., :self.args.input_dim]
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(input_data, tid, diw, distance_matrix=self.distance_matrix)
            
            if self.args.real_value:
                output = self.scaler.inverse_transform(output)

            loss = self.loss(output, label)
            loss.backward()

            # Gradient clipping
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            
            self.optimizer.step()
            total_loss += loss.item()

            # Log information
            if (batch_idx + 1) % self.args.log_step == 0:
                self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
                    epoch, batch_idx + 1, self.train_per_epoch, loss.item()))
                    
        train_epoch_loss = total_loss / self.train_per_epoch
        self.logger.info('********Train Epoch {}: averaged Loss: {:.6f}, time: {:.2f} s'.format(
            epoch, train_epoch_loss, time.time() - epoch_time))

        # Learning rate decay
        if self.args.lr_decay and self.lr_scheduler is not None:
            self.lr_scheduler.step()
            
        return train_epoch_loss

    def train(self):
        best_model = None
        best_test_model = None
        not_improved_count = 0
        best_loss = float('inf')
        best_test_loss = float('inf')
        
        for epoch in range(1, self.args.epochs + 1):
            train_epoch_loss = self.train_epoch(epoch)
            
            if self.val_loader == None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader
            test_dataloader = self.test_loader

            val_epoch_loss = self.val_epoch(epoch, val_dataloader)
            test_epoch_loss = self.test_epoch(epoch, test_dataloader)
            
            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break
                
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
                
            # Early stop
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn't improve for {} epochs. "
                                    "Training stops.".format(self.args.early_stop_patience))
                    break
                    
            # Save the best state
            if best_state == True:
                self.logger.info('*********************************Current best model saved!')
                best_model = copy.deepcopy(self.model.state_dict())

            if test_epoch_loss < best_test_loss:
                best_test_loss = test_epoch_loss
                best_test_model = copy.deepcopy(self.model.state_dict())

        # Save the best model to file
        if not self.args.debug:
            torch.save(best_model, self.best_path)
            self.logger.info("Saving current best model to " + self.best_path)
            torch.save(best_test_model, self.best_test_path)
            self.logger.info("Saving current best test model to " + self.best_test_path)

        # Test
        self.model.load_state_dict(best_model)
        self.test(self.model, self.args, self.test_loader, self.scaler, self.logger)

        self.logger.info("This is best_test_model")
        self.model.load_state_dict(best_test_model)
        self.test(self.model, self.args, self.test_loader, self.scaler, self.logger)

    @staticmethod
    def test(model, args, data_loader, scaler, logger, path=None):
        if path != None:
            check_point = torch.load(path)
            state_dict = check_point['state_dict']
            args = check_point['config']
            model.load_state_dict(state_dict)
            model.to(args.device)
            
        model.eval()
        y_pred = []
        y_true = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data.to(args.device)
                target = target.to(args.device)
                label = target[..., :args.output_dim]
                
                # Extract temporal information
                if data.shape[-1] > args.input_dim:
                    tid_data = data[..., args.input_dim:args.input_dim+1]
                    diw_data = data[..., -1:]
                    tid = (tid_data.squeeze(-1) * 287).long()
                    diw = diw_data.squeeze(-1).long()
                else:
                    tid = None
                    diw = None
                
                input_data = data[..., :args.input_dim]
                output = model(input_data, tid, diw, 
                             distance_matrix=getattr(args, 'distance_matrix', None))
                
                y_true.append(label)
                y_pred.append(output)

        if args.real_value:
            y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))
            y_true = torch.cat(y_true, dim=0)
        else:
            y_pred = torch.cat(y_pred, dim=0)
            y_true = torch.cat(y_true, dim=0)
            
        for t in range(y_true.shape[1]):
            mae, rmse, mape, _, pcc = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...],
                                                args.mae_thresh, args.mape_thresh)
            logger.info("Horizon {:02d}, RMSE: {:.4f}, MAE: {:.4f}, PCC: {:.4f}%".format(
                t + 1, rmse, mae, pcc*100))
                
        mae, rmse, mape, _, pcc = All_Metrics(y_pred, y_true, args.mae_thresh, args.mape_thresh)
        logger.info("Average Horizon, RMSE: {:.4f}, MAE: {:.4f}, PCC: {:.4f}%".format(
                    rmse, mae, pcc*100))
