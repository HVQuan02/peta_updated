# Define hyperparameters (adjust as needed)
batch_size = 32
num_epochs = 150
learning_rate = 2e-4
weight_decay = 1e-4
warmup_epochs = 5  # For learning rate scheduler

# Load dataset paths and class labels
train_data_dir, val_data_dir = get_train_val_split("path/to/data")
class_labels = ["event1", "event2", ...]  # Replace with actual class labels

# Create datasets and dataloaders
train_dataset = PhotoAlbumDataset(train_data_dir, class_labels)
val_dataset = PhotoAlbumDataset(val_data_dir, class_labels)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)