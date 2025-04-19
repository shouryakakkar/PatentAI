import torch
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, losses, InputExample
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from typing import List, Dict, Tuple
import json
import os
from datetime import datetime
from collections import defaultdict

class PatentDataset(Dataset):
    def __init__(self, examples: List[InputExample]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

class ModelTrainer:
    def __init__(self, base_model_name: str = 'all-MiniLM-L6-v2'):
        self.base_model_name = base_model_name
        self.model = SentenceTransformer(base_model_name)
        self.training_data = []
        self.validation_data = []
        self.training_stats = defaultdict(list)
        
        # Create directories for saving data and models
        os.makedirs('data/training', exist_ok=True)
        os.makedirs('models/checkpoints', exist_ok=True)

    def add_training_pair(self, text1: str, text2: str, similarity: float):
        """Add a pair of texts with their similarity score for training"""
        example = InputExample(texts=[text1, text2], label=similarity)
        self.training_data.append(example)

    def add_validation_pair(self, text1: str, text2: str, similarity: float):
        """Add a pair of texts with their similarity score for validation"""
        example = InputExample(texts=[text1, text2], label=similarity)
        self.validation_data.append(example)

    def collect_training_data_from_search(self, query: str, relevant_patents: List[Dict], irrelevant_patents: List[Dict]):
        """Collect training data from search results with user feedback"""
        # Add positive examples (relevant patents)
        for patent in relevant_patents:
            patent_text = f"{patent['title']} {patent['abstract']}"
            self.add_training_pair(query, patent_text, 1.0)
            
            # Add related patent pairs as positive examples
            for other_patent in relevant_patents:
                if patent != other_patent:
                    other_text = f"{other_patent['title']} {other_patent['abstract']}"
                    self.add_training_pair(patent_text, other_text, 0.8)

        # Add negative examples (irrelevant patents)
        for patent in irrelevant_patents:
            patent_text = f"{patent['title']} {patent['abstract']}"
            self.add_training_pair(query, patent_text, 0.0)

    def train(self, 
              batch_size: int = 16, 
              epochs: int = 4, 
              learning_rate: float = 2e-5,
              validation_split: float = 0.1):
        """Train the model on collected data"""
        if not self.training_data:
            raise ValueError("No training data available")

        # Split data into training and validation if validation_data is empty
        if not self.validation_data:
            split_idx = int(len(self.training_data) * (1 - validation_split))
            self.validation_data = self.training_data[split_idx:]
            self.training_data = self.training_data[:split_idx]

        # Create data loaders
        train_dataset = PatentDataset(self.training_data)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = PatentDataset(self.validation_data)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

        # Define the loss function
        train_loss = losses.CosineSimilarityLoss(self.model)

        # Train the model
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            optimizer_params={'lr': learning_rate},
            evaluation_steps=100,
            evaluator=None,  # We'll implement custom evaluation
            output_path=f'models/checkpoints/{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        )

        # Save training statistics
        self.save_training_stats()

    def evaluate(self) -> Dict:
        """Evaluate the model on validation data"""
        if not self.validation_data:
            raise ValueError("No validation data available")

        val_dataset = PatentDataset(self.validation_data)
        val_dataloader = DataLoader(val_dataset, batch_size=32)
        
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                texts = [example.texts for example in batch]
                labels = torch.tensor([example.label for example in batch])
                
                embeddings1 = self.model.encode([t[0] for t in texts])
                embeddings2 = self.model.encode([t[1] for t in texts])
                
                similarities = torch.cosine_similarity(
                    torch.tensor(embeddings1),
                    torch.tensor(embeddings2)
                )
                
                # Convert similarities to binary predictions (threshold = 0.5)
                predictions = (similarities > 0.5).float()
                correct_predictions += (predictions == labels).sum().item()
                
                # Calculate loss
                total_loss += torch.nn.functional.mse_loss(similarities, labels).item()

        accuracy = correct_predictions / len(self.validation_data)
        avg_loss = total_loss / len(val_dataloader)

        return {
            'accuracy': accuracy,
            'average_loss': avg_loss
        }

    def save_training_stats(self):
        """Save training statistics to a file"""
        stats = {
            'model_name': self.base_model_name,
            'training_data_size': len(self.training_data),
            'validation_data_size': len(self.validation_data),
            'training_stats': dict(self.training_stats),
            'timestamp': datetime.now().isoformat()
        }
        
        filename = f'data/training/stats_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(filename, 'w') as f:
            json.dump(stats, f, indent=2)

    def save_model(self, path: str):
        """Save the trained model"""
        self.model.save(path)

    def load_model(self, path: str):
        """Load a trained model"""
        self.model = SentenceTransformer(path)

# Function to create training pairs from patent data
def create_training_pairs(patents: List[Dict]) -> List[Tuple[str, str, float]]:
    """Create training pairs from a list of patents"""
    training_pairs = []
    
    for i, patent1 in enumerate(patents):
        text1 = f"{patent1['title']} {patent1['abstract']}"
        
        # Create pairs with other patents
        for j, patent2 in enumerate(patents[i+1:]):
            text2 = f"{patent2['title']} {patent2['abstract']}"
            
            # Calculate similarity based on IPC codes
            same_ipc = any(c1['section'] == c2['section'] 
                          for c1 in patent1.get('predicted_classes', [])
                          for c2 in patent2.get('predicted_classes', []))
            
            # Assign similarity score
            similarity = 0.8 if same_ipc else 0.2
            
            training_pairs.append((text1, text2, similarity))
    
    return training_pairs 