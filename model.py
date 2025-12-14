import torch.nn as nn
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from autoencoder import Autoencoder
from evaluation import Evaluation
from mlp import MLP
from utils import MODEL_FOLDER


class CellDrugAttention(nn.Module):
    def __init__(self, d_cell, d_drug, d_model, n_heads, d_ff=1024, dropout=0.2):
        super().__init__()

        self.q_proj_cell = nn.Linear(d_cell, d_model)
        self.k_proj_drug = nn.Linear(d_drug, d_model)
        self.v_proj_drug = nn.Linear(d_drug, d_model)

        self.q_proj_drug = nn.Linear(d_drug, d_model)
        self.k_proj_cell = nn.Linear(d_cell, d_model)
        self.v_proj_cell = nn.Linear(d_cell, d_model)

        self.attn_cell2drug = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.attn_drug2cell = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.ff1 = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.norm3 = nn.LayerNorm(d_model)
        self.ff2 = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm4 = nn.LayerNorm(d_model)

        self.fc = nn.Linear(2 * d_model + d_cell + d_drug, d_model)

    def forward(self, zc, zd):
        q_cell = self.q_proj_cell(zc).unsqueeze(1)
        k_drug = self.k_proj_drug(zd).unsqueeze(1)
        v_drug = self.v_proj_drug(zd).unsqueeze(1)

        attn_cell2drug, _ = self.attn_cell2drug(q_cell, k_drug, v_drug)
        x = self.norm1(q_cell + attn_cell2drug)
        ff_out1 = self.ff1(x)
        x = self.norm2(x + ff_out1)

        q_drug = self.q_proj_drug(zd).unsqueeze(1)
        k_cell = self.k_proj_cell(zc).unsqueeze(1)
        v_cell = self.v_proj_cell(zc).unsqueeze(1)

        attn_drug2cell, _ = self.attn_drug2cell(q_drug, k_cell, v_cell)
        y = self.norm3(q_drug + attn_drug2cell)
        ff_out2 = self.ff2(y)
        y = self.norm4(y + ff_out2)

        cell_feature = torch.cat([x.squeeze(1), zc], dim=1)
        
        drug_feature = torch.cat([y.squeeze(1), zd], dim=1)

        fused_features = torch.cat([cell_feature, drug_feature], dim=1)

        final_out = self.fc(fused_features)

        return final_out


class Model(nn.Module):
    def __init__(self, cell_modality_sizes, drug_modality_sizes, cell_ae_latent_dim, drug_ae_latent_dim):
        super(Model, self).__init__()

        self.cell_autoencoder = Autoencoder(sum(cell_modality_sizes), cell_ae_latent_dim)
        self.drug_autoencoder = Autoencoder(sum(drug_modality_sizes), drug_ae_latent_dim)
        
        d_model = 256
        n_heads = 8
        self.attn_module = CellDrugAttention(cell_ae_latent_dim, drug_ae_latent_dim, d_model=d_model, n_heads=n_heads)
        
        attn_latent_dim = d_model

        self.mlp = MLP(attn_latent_dim, 1)

    def forward(self, cell_x, drug_x):
        cell_encoded = self.cell_autoencoder.encoder(cell_x)
        cell_decoded = self.cell_autoencoder.decoder(cell_encoded)

        drug_encoded = self.drug_autoencoder.encoder(drug_x)
        drug_decoded = self.drug_autoencoder.decoder(drug_encoded)
        
        attn_output = self.attn_module(cell_encoded, drug_encoded)

        mlp_output = self.mlp(attn_output)

        return cell_decoded, drug_decoded, mlp_output



def train(model, train_loader, val_loader,  num_epochs,class_weights):
   
    autoencoder_loss_fn = nn.MSELoss()
    mlp_loss_fn = nn.BCELoss()

    train_accuracies = []
    val_accuracies = []

    train_loss = []
    val_loss = []

    mlp_optimizer = optim.Adam(model.parameters(), lr=0.0005,)
    scheduler = lr_scheduler.ReduceLROnPlateau(mlp_optimizer, mode='min', factor=0.8, patience=5)

    cell_ae_weight = 1.0
    drug_ae_weight = 1.0
    mlp_weight = 1.0

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        train_correct = 0
        train_total_samples = 0
        for batch_idx, (cell_data, drug_data, target) in enumerate(train_loader):
            mlp_optimizer.zero_grad()

            cell_decoded_output, drug_decoded_output, mlp_output = model(cell_data, drug_data)

            cell_ae_loss = cell_ae_weight * autoencoder_loss_fn(cell_decoded_output, cell_data)
            drug_ae_loss = drug_ae_weight * autoencoder_loss_fn(drug_decoded_output, drug_data)
            mlp_loss = mlp_weight * mlp_loss_fn(mlp_output, target)

            total_loss = drug_ae_loss + cell_ae_loss + mlp_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            mlp_optimizer.step()
            total_train_loss += total_loss.item()

            train_predictions = torch.round(mlp_output)
            train_correct += (train_predictions == target).sum().item()
            train_total_samples += target.size(0)

        avg_train_loss = total_train_loss / len(train_loader)
        train_loss.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0.0
        correct = 0
        total_samples = 0
        with torch.no_grad():
            for val_batch_idx, (cell_data_val, drug_data_val, val_target) in enumerate(val_loader):
                cell_decoded_output_val, drug_decoded_output_val, mlp_output_val = model(cell_data_val, drug_data_val)
                
                cell_ae_loss_val = cell_ae_weight * autoencoder_loss_fn(cell_decoded_output_val, cell_data_val)
                drug_ae_loss_val = drug_ae_weight * autoencoder_loss_fn(drug_decoded_output_val, drug_data_val)
                mlp_loss_val = mlp_weight * mlp_loss_fn(mlp_output_val, val_target)

                total_val_loss = (drug_ae_loss_val + cell_ae_loss_val + mlp_loss_val).item()
                
                val_predictions = torch.round(mlp_output_val)
                correct += (val_predictions == val_target).sum().item()
                total_samples += val_target.size(0)

            avg_val_loss = total_val_loss / len(val_loader)
            val_loss.append(avg_val_loss)


        train_accuracy = train_correct / train_total_samples
        train_accuracies.append(train_accuracy)
        val_accuracy = correct / total_samples
        val_accuracies.append(val_accuracy)

        print(
            'Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}, Train Accuracy: {:.4f}, Val Accuracy: {:.4f}'.format(
                epoch + 1, num_epochs, avg_train_loss, avg_val_loss, train_accuracy,
                val_accuracy))

        scheduler.step(total_train_loss)

    torch.save(model.state_dict(), MODEL_FOLDER + 'MODEL.pth')


def test(model, test_loader, reverse=False):
   
    model.eval()

    all_predictions = []
    all_labels = []

    for i, (test_cell_loader, test_drug_loader, labels) in enumerate(test_loader):
        with torch.no_grad():
            decoded_cell_output, decoded_drug_output, mlp_output = model(test_cell_loader, test_drug_loader)

        predictions = 1 - mlp_output if reverse else mlp_output

    result = Evaluation.evaluate(labels, predictions)

    return result

