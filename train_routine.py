from torch.utils.data import DataLoader
import torch.optim as optim

class TrainRoutine:
    def __init__(self, device, model, dataset, loss_function, optimizer):
        self.device = device
        self.model = model
        self.dataset = dataset
        self.loss_function = loss_function
        self.optimizer = optimizer


    def fit(self , num_epoch):
        for epoch_index in range(num_epoch):
            print(epoch_index)
            dataloader = self.generate_batches(self.dataset, 4, device = self.device)
            self.model.train()
            for batch_index, (X, y) in enumerate(dataloader):
                print(batch_index)
                #resetting gradients
                self.optimizer.zero_grad()
                #model step forward
                y_pred = model(X)
                #calc loss function
                loss = self.loss_function(y_pred, y)
                #backward step
                loss.backward()
                #propagate gradient
                self.optimizer.step()

        return self.device

    def generate_batches(self, dataset, batch_size, shuffle=True,
                        drop_last=True, device="cpu"):
        """
        A generator function which wraps the PyTorch DataLoader. It will 
        ensure each tensor is on the write device location.
        """
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                                shuffle=shuffle, drop_last=drop_last)

        return dataloader


