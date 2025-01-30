import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import nibabel
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ToTensord, Compose, RandFlipd, CenterSpatialCropd
)
from monai.data import DataLoader, Dataset, pad_list_data_collate
import torch.nn as nn
import torch.optim as optim
import time


# Load and preprocess metadata
#data_csv = "/work/souza_lab/Aminreza/thesis_data/data_aggregated.csv"  
#data_dir = "/work/souza_lab/Aminreza/thesis_data"
data_csv = "/Users/aminrezaabbasi/Desktop/Dataset/data_aggregated.csv"  
data_dir = "/Users/aminrezaabbasi/Desktop/Dataset"
df = pd.read_csv(data_csv)

# Add file paths to the dataframe
df["file_path"] = df["file_name"].apply(lambda x: os.path.join(data_dir, x))

# Verify file existence
missing_files = [path for path in df["file_path"] if not os.path.exists(path)]
if missing_files:
    raise FileNotFoundError(f"The following files are missing: {missing_files}")

# Bin the Age column into ranges
bins = [0, 20, 30, 40, 50, 60, 70, 80, 100]  # Adjust bins as needed
labels = ['0-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80+']
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)

# Create the stratify_key column
df["stratify_key"] = df["AgeGroup"].astype(str) + "-" + df["Sex"] + "-" + df["dataset_name"]

# Count the occurrences of each stratify key
key_counts = df["stratify_key"].value_counts()

# Identify keys with fewer samples than required for stratification
min_samples_per_group = 10  # Adjust based on dataset size and split ratio
rare_keys = key_counts[key_counts < min_samples_per_group].index

# Merge rare keys into a single category
df["stratify_key"] = df["stratify_key"].apply(lambda x: "rare" if x in rare_keys else x)

try:
    # Perform stratified splitting
    train_df, temp_df = train_test_split(
        df, test_size=0.3, stratify=df["stratify_key"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=2 / 3, stratify=temp_df["stratify_key"], random_state=42
    )

    # Drop unnecessary columns
    train_df.drop(columns=["stratify_key", "AgeGroup"], inplace=True)
    val_df.drop(columns=["stratify_key", "AgeGroup"], inplace=True)
    test_df.drop(columns=["stratify_key", "AgeGroup"], inplace=True)

    print("Data split completed:")
    print(f"Training set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")

except ValueError as e:
    print(f"Stratified split failed after merging rare keys: {e}")
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=2 / 3, random_state=42)

# Check the distribution for each set
def check_distribution(df, label):
    print(f"Distribution in {label}:")
    print("\nAgeGroup:")
    print(df['AgeGroup'].value_counts(normalize=True))
    print("\nSex:")
    print(df['Sex'].value_counts(normalize=True))
    print("\nDataset Name:")
    print(df['dataset_name'].value_counts(normalize=True))
    print("-" * 50)

# Add AgeGroup back to analyze the split
bins = [0, 20, 30, 40, 50, 60, 70, 80, 100]
labels = ['0-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80+']
for df_set, name in zip([train_df, val_df, test_df], ['Training', 'Validation', 'Test']):
    df_set['AgeGroup'] = pd.cut(df_set['Age'], bins=bins, labels=labels)
    check_distribution(df_set, name)

# Define MONAI Transforms for dictionary-based operations
transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    ScaleIntensityd(keys=["image"]),
    CenterSpatialCropd(keys=["image"], roi_size=(256, 256, 256)),  # Crop to fixed size
    RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
    ToTensord(keys=["image", "label"])  # Convert both image and label to tensors
])

train_dataset = Dataset(
    data=[{"image": row.file_path, "label": float(row.Age)} for _, row in train_df.iterrows()],
    transform=transforms
)

val_dataset = Dataset(
    data=[{"image": row.file_path, "label": float(row.Age)} for _, row in val_df.iterrows()],
    transform=transforms
)

test_dataset = Dataset(
    data=[{"image": row.file_path, "label": float(row.Age)} for _, row in test_df.iterrows()],
    transform=transforms
)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=pad_list_data_collate, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=pad_list_data_collate, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=pad_list_data_collate, num_workers=4, pin_memory=True)

# Define the CNN model
class EnhancedCNN(nn.Module):
    def __init__(self):
        super(EnhancedCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.global_pool(x).squeeze()
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EnhancedCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with model saving
def train_model(epochs, model_save_path="brain_age_model.pth"):
    best_val_loss = float("inf")
    for epoch in range(epochs):
        train_loss = 0.0
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            start_time = time.time()
            images, labels = batch["image"].to(device), batch["label"].to(device)
            end_time = time.time()
            data_loading_time = end_time - start_time
            # Print the data loading time
            optimizer.zero_grad()
            outputs = model(images)
            outputs = outputs.view(-1)
            labels = labels.view(-1).float()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(train_loader)} - Data Loading Time: {data_loading_time:.4f} seconds")

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                images, labels = batch["image"].to(device), batch["label"].to(device)
                outputs = model(images)
                loss = criterion(outputs.view(-1), labels.float())
                val_loss += loss.item()
        
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_loader):.4f}, "
              f"Validation Loss: {val_loss / len(val_loader):.4f}")

        # Save the model after every epoch
        epoch_model_path = f"epoch_{epoch + 1}_model.pth"
        torch.save(model.state_dict(), epoch_model_path)
        print(f"Model saved to {epoch_model_path}")
        print(f"Total Evaluation Time: {end_time - start_time:.2f}s")


        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Best model updated and saved to {model_save_path}")

    print("Training complete")

# Evaluate the model
def evaluate_model(model_load_path="brain_age_model.pth"):
    model.load_state_dict(torch.load(model_load_path))
    model.eval()
    test_loss = 0.0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch_start_time = time.time()  # Start time for batch
            images, labels = batch["image"].to(device), batch["label"].to(device)
            outputs = model(images)
            loss = criterion(outputs.view(-1), labels.float())
            test_loss += loss.item()
            predictions.extend(outputs.view(-1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

            batch_end_time = time.time()  # End time for batch
            print(f"Batch {batch_idx + 1}/{len(test_loader)}: "
                  f"Batch Time: {batch_end_time - batch_start_time:.2f}s")
            
    end_time = time.time()  # End time for evaluation

    print(f"Test Loss: {test_loss / len(test_loader):.4f}")

    results_df = pd.DataFrame({"True_Age": true_labels, "Predicted_Age": predictions})
    results_df.to_csv("test_results.csv", index=False)
    print("Results saved to test_results.csv")

# Run training and evaluation
train_model(epochs=15)
evaluate_model()

if __name__ == "__main__":
    # Run training and evaluation
    train_model(epochs=15)
    evaluate_model()
