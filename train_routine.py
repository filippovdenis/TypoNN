from torch.utils.data import DataLoader

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
            batch_generator = self.generate_batches(self.dataset, 4, device = self.device)
            self.model.train()
            for batch_index, batch_dict in enumerate(batch_generator):
                print(batch_index)
        return self.device

    def generate_batches(self, dataset, batch_size, shuffle=True,
                        drop_last=True, device="cpu"):
        """
        A generator function which wraps the PyTorch DataLoader. It will 
        ensure each tensor is on the write device location.
        """
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                                shuffle=shuffle, drop_last=drop_last)

        for data_dict in dataloader:
            out_data_dict = {}
            for name, tensor in data_dict.items():
                out_data_dict[name] = data_dict[name].to(device)
            yield out_data_dict


