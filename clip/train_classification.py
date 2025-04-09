import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle

import numpy as np
from tqdm import tqdm

if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


# Custom Dataset Class
class EmbeddingsDataset(Dataset):
    def __init__(self, embeddings_path, labels_file, cats_indexes_file):
        self.embeddings_path = embeddings_path
        with open(labels_file, "r") as file:
            self.labels = json.load(file)
        self.names = list(self.labels.keys())
        self.cats_indexes = None
        self.read_cats_map(cats_indexes_file)

    def read_cats_map(self, fpath):
        with open(fpath, "r") as file:
            cats_indexes = json.load(file)

        self.cats_indexes = {item[1]: item[0] for item in cats_indexes}

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]

        with open("{}/{}.pkl".format(self.embeddings_path, name), "rb") as file:
            embedding = torch.tensor(pickle.load(file))

        label = self.cats_indexes[self.labels[name]]

        return embedding, label


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def main():
    # exp = "t5"
    exp = "clip_base"
    # Hyperparameters
    input_dim = 512
    hidden_dim = 256
    output_dim = 23
    batch_size = 4098
    learning_rate = 0.0005
    epochs = 20

    ds_path = "/home/docker_user/work/clip/data/classification"
    # train_embeddings_path = "{}/train_{}".format(ds_path, exp)
    train_embeddings_path = "{}/train_{}_text".format(ds_path, exp)
    train_labels_file = "{}/train_classification_2017.json".format(ds_path)
    # train_embeddings_path = "{}/val_{}".format(ds_path, exp)
    # train_labels_file = "{}/val_classification_2017.json".format(ds_path)
    # val_embeddings_path = "{}/val_{}".format(ds_path, exp)
    val_embeddings_path = "{}/val_{}_text".format(ds_path, exp)
    val_labels_file = "{}/val_classification_2017.json".format(ds_path)
    cats_map_path = "{}/materials_general_classification.json".format(ds_path)

    # Create custom Dataset for training and validation
    train_dataset = EmbeddingsDataset(
        train_embeddings_path, train_labels_file, cats_map_path
    )
    val_dataset = EmbeddingsDataset(val_embeddings_path, val_labels_file, cats_map_path)

    # Create DataLoader for batching
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_acc = 0
    # Train the model
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in tqdm(train_loader):
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # For accuracy calculation
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total * 100

        # Validation step
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = val_correct / val_total * 100

        if val_acc > best_acc:
            best_acc = val_acc
            print("Saving weights")
            torch.save(model.state_dict(), f"mlp_mat_cls_{exp}_text.pth")

        print(
            f"Epoch {epoch+1}/{epochs} - "
            f"Training Loss: {train_loss:.4f}, "
            f"Training Accuracy: {train_acc:.2f}%, "
            f"Validation Accuracy: {val_acc:.2f}%"
        )

    print(f"Best accuracy: {best_acc}")
    # Save the trained model (optional)
    # torch.save(model.state_dict(), 'mlp_model.pth')


if __name__ == "__main__":
    main()
