import torch
import torch.nn as nn
import numpy as np

class FrequencyWeightedMAE(nn.Module):
    def __init__(self, beta=0.999, bins=None):
        super(FrequencyWeightedMAE, self).__init__()
        self.beta = beta
        # Physical bins in µg/m³ for NCAP / AQI thresholds
        if bins is None:
            self.bins = [-float('inf'), 30.0, 60.0, 90.0, 120.0, 250.0, 400.0, float('inf')]
        else:
            self.bins = bins

    def forward(self, y_pred, y_true):
        # Calculate standard absolute error
        l1_loss = torch.abs(y_pred - y_true)

        # Vectorized frequency weighting based on true Z-score values
        weights = torch.ones_like(y_true)
        y_true_np = y_true.detach().cpu().numpy()
        
        # Compute frequencies in the current batch
        hist, _ = np.histogram(y_true_np, bins=self.bins)
        
        # Avoid zero division
        hist = np.where(hist == 0, 1, hist) 
        
        # Calculate effective number of samples weights: (1 - beta) / (1 - beta^n)
        effective_num = (1.0 - np.power(self.beta, hist))
        bin_weights = (1.0 - self.beta) / effective_num
        
        # Normalize weights so they scale cleanly with learning rate
        bin_weights = bin_weights / np.sum(bin_weights) * len(self.bins)
        
        # Apply the computed weights map to the tensors
        for i in range(len(self.bins) - 1):
            mask = (y_true >= self.bins[i]) & (y_true < self.bins[i+1])
            weights[mask] = bin_weights[i]

        weighted_loss = torch.mean(weights * l1_loss)
        
        # CUSTOM VARIANCE PENALTY: Force model to match the amplitude of the ground truth curve
        pred_std = torch.std(y_pred, dim=-1)
        true_std = torch.std(y_true, dim=-1)
        variance_penalty = torch.mean(torch.abs(pred_std - true_std))
        
        # Combine losses (0.7 fMAE + 0.3 Variance)
        total_loss = weighted_loss + 0.3 * variance_penalty
        
        return total_loss
