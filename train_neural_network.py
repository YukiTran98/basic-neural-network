from dataset import MyDataset
from basic_neural_network import SimpleNeuralNetwork
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    
    # Define epoch
    num_epoch = 5

    # Get dataset
    train_dataset = MyDataset(root="./cifar/cifar-10-batches-py", train=True)
    train_dataloader = DataLoader(
        dataset = train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    test_dataset = MyDataset(root="./cifar/cifar-10-batches-py", train=False)
    test_dataset = DataLoader(
        dataset = test_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    model = SimpleNeuralNetwork(num_class=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    num_iters = len(train_dataloader)

    # loop
    for epoch in range(num_epoch):
        model.train()
        for iter, (images, labels) in enumerate(train_dataloader):
            # Forward
            outputs = model(images) # forward
            loss = criterion(outputs, labels)
            # print(f"Epoch {epoch+1}/{num_epoch}. Iteration {iter+1}/{num_iters}. Loss {loss}")


            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() # Update parameter

        model.eval()
        all_predictions = []
        all_labels = []
        for iter, (images, labels) in enumerate(test_dataset):
            all_labels.extend(labels)
            with torch.no_grad():
                predictions = model(images)
                indices = torch.argmax(predictions, dim=1)
                all_predictions.extend(indices)
                loss = criterion(predictions, labels)
        
        all_labels = [label.item() for label in all_labels]
        all_predictions = [label.item() for label in all_predictions]

        print(f"Epoch {epoch+1}")
        print(accuracy_score(all_labels, all_predictions))


