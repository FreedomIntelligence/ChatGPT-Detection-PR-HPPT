import json

import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)


# Definition of a custom dataset for the sequence classification task
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, labels, tokenizer, max_length=512):
        self.inputs = inputs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    # Return the number of examples in the dataset
    def __len__(self):
        return len(self.inputs)

    # Return a single example and its corresponding label
    def __getitem__(self, index):
        input_ids = self.tokenizer.encode(
            self.inputs[index], add_special_tokens=True, max_length=self.max_length
        )
        label = self.labels[index]
        return input_ids, label


# Definition of a model trainer for the sequence classification task
class ModelTrainer:
    def __init__(
        self,
        train_file,
        val_file,
        test_file,
        model_path=None,
        model_name="roberta-base",
        batch_size=16,
        num_epochs=10,
        learning_rate=2e-5,
        warmup_steps=0.1,
    ):
        # Load the training, validation, and test data from JSON files
        self.train_data = self.load_data(train_file)
        self.val_data = self.load_data(val_file)
        self.test_data = self.load_data(test_file)
        # Instantiate a tokenizer and a pre-trained model for sequence classification
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaForSequenceClassification.from_pretrained(model_name).cuda()
        # Set the batch size, number of epochs, loss function, optimizer, and learning rate scheduler
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(warmup_steps * len(self.train_data) / self.batch_size),
            num_training_steps=len(self.train_data)
            * self.num_epochs
            // self.batch_size,
        )

    # Load data from a JSON file and return a list of examples
    def load_data(self, filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
        examples = []
        for example in data:
            text = example["text"]
            label = example["fake"]
            examples.append({"text": text, "label": label})
        return examples

    # Tokenize the inputs and labels and return them as two lists
    def tokenize_inputs(self, data):
        inputs = []
        labels = []
        for example in data:
            input_ids = self.tokenizer.encode(
                example["text"],
                add_special_tokens=True,
                truncation=True,
                max_length=512,
            )
            inputs.append(input_ids)
            labels.append(example["label"])
        return inputs, labels

    # Function to pad input sequences and return them in a batch
    def collate_fn(self, batch):
        input_ids = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        # Get the maximum length of the input sequences in the batch
        max_length = max(len(ids) for ids in input_ids)
        input_ids_padded = []
        attention_masks = []
        # Pad the input sequences and create attention masks
        for ids in input_ids:
            padding = [0] * (max_length - len(ids))
            input_ids_padded.append(ids + padding)
            attention_masks.append([1] * len(ids) + padding)
        # Return the inputs and labels as a dictionary and a tensor, respectively
        inputs = {
            "input_ids": torch.tensor(input_ids_padded),
            "attention_mask": torch.tensor(attention_masks),
        }
        return inputs, torch.tensor(labels)

    # Train the model on a given dataloader
    def train_model(self, train_loader):
        # Set the model to training mode and initialize the total loss
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(total=len(train_loader))
        # Iterate over the batches in the dataloader
        for step, (inputs, labels) in enumerate(train_loader, start=1):
            # Clear the gradients, get the model outputs, and calculate the loss
            self.optimizer.zero_grad()
            outputs = self.model(
                inputs["input_ids"].cuda(),
                attention_mask=inputs["attention_mask"].cuda(),
            )
            loss = self.criterion(outputs.logits, labels.cuda())
            # Backpropagate the loss, update the parameters, and adjust the learning rate
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            # Update the total loss and calculate the average loss
            total_loss += loss.item()
            avg_loss = total_loss / step
            # Update the progress bar
            pbar.set_description(f"avg_loss: {avg_loss:.4f}")
            pbar.update(1)
        pbar.close()
        # Return the average loss over the entire dataset
        return total_loss / len(train_loader)

    # Evaluate the model on a given dataloader
    def evaluate_model(self, test_loader):
        # Set the model to evaluation mode and initialize the true and predicted labels
        self.model.eval()
        true_labels = []
        predicted_labels = []
        with torch.no_grad():
            # Iterate over the batches in the dataloader and get the model outputs
            for inputs, labels in test_loader:
                outputs = self.model(
                    inputs["input_ids"].cuda(),
                    attention_mask=inputs["attention_mask"].cuda(),
                )
                # Append the true and predicted labels to their respective lists
                true_labels.extend(labels)
                predicted_labels.extend(torch.argmax(outputs.logits, dim=1).cpu())
        # Calculate the classification report and the accuracy of the model
        report = classification_report(true_labels, predicted_labels, digits=4)
        return (
            report,
            torch.sum(torch.tensor(true_labels) == torch.tensor(predicted_labels))
            / len(true_labels),
        )

    # Train and evaluate the model for a given number of epochs
    def run_training(self):
        # Tokenize the inputs and labels for the training, validation, and test datasets
        train_inputs, train_labels = self.tokenize_inputs(self.train_data)
        val_inputs, val_labels = self.tokenize_inputs(self.val_data)
        test_inputs, test_labels = self.tokenize_inputs(self.test_data)

        # Create dataloaders for the training, validation, and test datasets
        train_dataset = CustomDataset(train_inputs, train_labels, self.tokenizer)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )
        val_dataset = CustomDataset(val_inputs, val_labels, self.tokenizer)
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn
        )
        test_dataset = CustomDataset(test_inputs, test_labels, self.tokenizer)
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn
        )

        # Train the model for a given number of epochs and save the best model based on the validation accuracy
        best_accuracy = 0
        for epoch in range(self.num_epochs):
            train_loss = self.train_model(train_loader)
            val_report, val_accuracy = self.evaluate_model(val_loader)
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save(self.model.state_dict(), "best_model.pt")
            # Print the epoch number, training loss, and validation accuracy
            print(
                f"Epoch {epoch + 1}, train loss: {train_loss:.4f}, val accuracy: {val_accuracy:.4f}"
            )
            # Print the classification report for the validation dataset
            print(val_report)
        # Load the best model based on the validation accuracy and evaluate it on the test dataset
        self.model.load_state_dict(torch.load("best_model.pt"))
        test_report, test_accuracy = self.evaluate_model(test_loader)
        # Print the best accuracy and the classification report for the test dataset
        print(f"Best accuracy: {test_accuracy:.4f}")
        print(test_report)


if __name__ == "__main__":
    trainer = ModelTrainer(
        "../Dataset/HPPT/train.json", "../Dataset/HPPT/val.json", "../Dataset/HPPT/test.json", 
        model_path = "best_model.pt"
    )
    trainer.run_training()
