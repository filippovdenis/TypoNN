class TrainRoutine:
    def __init__(self, device, model, dataset, batch_generator, loss_function, optimizer):
        self.device = device
        self.model = model
        self.dataset = dataset
        self.batch_generator = batch_generator
        self.loss_function = loss_function
        self.optimizer = optimizer


    def fit(num_epoch):
        for epoch_index in range(num_epoch):
            print(epoch_index)
        return self.device


