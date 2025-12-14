from torch import nn
from torch import optim, no_grad
import torch
from evaluation import Evaluation
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.mlp(x)


def train_mlp(model, train_loader, val_loader, num_epochs):
    mlp_loss_fn = nn.BCELoss()

    mlp_optimizer = optim.Adadelta(model.parameters(), lr=0.01,)
    # scheduler = lr_scheduler.ReduceLROnPlateau(mlp_optimizer, mode='min', factor=0.8, patience=5, verbose=True)
    train_accuracies = []
    val_accuracies = []

    train_loss = []
    val_loss = []
    early_stopper = EarlyStopper(patience=3, min_delta=0.05)

    for epoch in range(num_epochs):
        # Training
        model.trainCombinedModel()
        total_train_loss = 0.0
        train_correct = 0
        train_total_samples = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            mlp_optimizer.zero_grad()
            mlp_output = model(data)

            mlp_loss = mlp_loss_fn(mlp_output, target)

            mlp_loss.backward()
            mlp_optimizer.step()

            total_train_loss += mlp_loss.item()

            # Calculate accuracy
            train_predictions = torch.round(mlp_output)
            train_correct += (train_predictions == target).sum().item()
            train_total_samples += target.size(0)

            # if batch_idx % 200 == 0:
            #     after_lr = mlp_optimizer.param_groups[0]["lr"]
            #     print('Epoch [{}/{}], Batch [{}/{}], Total Loss: {:.4f}, Learning Rate: {:.8f},'.format(
            #         epoch + 1, num_epochs, batch_idx + 1, len(train_loader), mlp_loss.item(), after_lr))

        avg_train_loss = total_train_loss / len(train_loader)
        train_loss.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0.0
        correct = 0
        total_samples = 0
        with torch.no_grad():
            for val_batch_idx, (data, val_target) in enumerate(val_loader):
                val_mlp_output = model(data)

                val_mlp_loss = mlp_loss_fn(val_mlp_output, val_target)

                total_val_loss += val_mlp_loss.item()

                # Calculate accuracy
                val_predictions = torch.round(val_mlp_output)
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
       
        before_lr = mlp_optimizer.param_groups[0]["lr"]
        # scheduler.step(avg_val_loss)
        after_lr = mlp_optimizer.param_groups[0]["lr"]
        if before_lr != after_lr:
            print("Epoch %d: Adam lr %.8f -> %.8f" % (epoch, before_lr, after_lr))

    Evaluation.plot_train_val_accuracy(train_accuracies, val_accuracies, epoch+1)
    Evaluation.plot_train_val_loss(train_loss, val_loss, epoch+1)


def test_mlp(model, test_loader):
    for i, (data, labels) in enumerate(test_loader):
        model.eval()
        with torch.no_grad():
            mlp_output = model(data)
        return Evaluation.evaluate(labels, mlp_output)

