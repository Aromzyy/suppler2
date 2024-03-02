import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
import pandas as pd



# Define global mean and standard deviation constants used for normalization
means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]

# Load CSV file
csv_file = '/Users/aromaatieno/Downloads/Capstone_Suppler_App 2/data_science/DDI_Dataset/filtered_data.csv'
data = pd.read_csv(csv_file)

# Extract unique disease labels from the 'disease' column
unique_disease_labels = data['disease'].unique()

# Create a mapping of disease names to indices
disease_to_index = {label: index for index, label in enumerate(unique_disease_labels)}
index_to_disease = {index: disease for disease, index in disease_to_index.items()}
# Function to get the model
def get_model(num_classes):
    model = mobilenet_v2(pretrained=False)
    # Replace the classifier with a new one for the correct number of classes
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

# the following code only runs when the script is executed directly, not when imported
if __name__ == "__main__":
    
    pass
