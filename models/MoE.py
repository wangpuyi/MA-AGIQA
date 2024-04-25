import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureTransform(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeatureTransform, self).__init__()
        self.norm = nn.BatchNorm1d(input_dim)
        self.fc = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        x = self.norm(x)
        x = F.relu(self.fc(x))
        return x

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return x

class MoENetwork(nn.Module):
    def __init__(self, cnn_dim, mllm_dim, hidden_dim, num_experts, dropout_rate):
        super(MoENetwork, self).__init__()
        # Each feature transform brings the input to a common hidden dimension
        self.cnn_transform = FeatureTransform(cnn_dim, hidden_dim)
        self.mllm_transform = FeatureTransform(mllm_dim, hidden_dim)
        
        self.experts = nn.ModuleList([Expert(hidden_dim*num_experts, hidden_dim, dropout_rate) for _ in range(num_experts)])
        
        self.gating = GatingNetwork(hidden_dim*num_experts, num_experts)
        
        self.output_fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, cnn_feature, mllm_feature1, mllm_feature2):
        # Transform features to common dimension
        transformed_cnn = self.cnn_transform(cnn_feature) # (batch_size, hidden_dim)
        transformed_mllm1 = self.mllm_transform(mllm_feature1)
        transformed_mllm2 = self.mllm_transform(mllm_feature2)
        
        concatenated_features = torch.cat((transformed_cnn, transformed_mllm1, transformed_mllm2), dim=1) # (batch_size, 3*hidden_dim)
        
        expert_outputs = [expert(concatenated_features) for expert in self.experts] # List of (batch_size, hidden_dim)
    
        stacked_experts = torch.stack(expert_outputs, dim=1) # (batch_size, num_experts, hidden_dim)
        gating_weights = self.gating(concatenated_features) # (batch_size, num_experts)
        
        # Weighted sum of expert outputs
        weighted_expert_outputs = torch.bmm(gating_weights.unsqueeze(1), stacked_experts).squeeze(1) # (batch_size, 1, num_experts) x (batch_size, num_experts, hidden_dim) -> (batch_size, hidden_dim)
        
        # Output layer to get the final score
        output_score = self.output_fc(weighted_expert_outputs)
        return output_score

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)
    
    def forward(self, x):
        return F.softmax(self.fc(x), dim=1)


if __name__ == '__main__':
    # Parameters for the MoE network
    batch_size = 32  # Batch size
    cnn_dim = 786  # Dimension of the CNN feature vector
    mllm_dim = 4096  # Dimension of the MLLM feature vectors
    hidden_dim = 512  # Common hidden dimension to which all inputs will be transformed
    num_experts = 3  # Number of experts in the MoE model
    dropout_rate = 0.5  # Dropout rate to be used in the expert networks

    # Instantiate the MoE model
    moe_model = MoENetwork(cnn_dim, mllm_dim, hidden_dim, num_experts, dropout_rate)

    # Example input tensors
    cnn_feature = torch.randn(batch_size, cnn_dim)
    mllm_feature1 = torch.randn(batch_size, mllm_dim)
    mllm_feature2 = torch.randn(batch_size, mllm_dim)

    # Forward pass to get the output score
    output_score = moe_model(cnn_feature, mllm_feature1, mllm_feature2)
