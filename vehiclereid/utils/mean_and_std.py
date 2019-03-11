import torch


def get_mean_and_std(dataloader, dataset):
    # Compute the mean and std value of dataset.
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, _, _ in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def calculate_mean_and_std(dataset_loader, dataset_size):
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for data in dataset_loader:
        now_batch_size, c, h, w = data[0].shape
        mean += torch.sum(torch.mean(torch.mean(data[0], dim=3), dim=2), dim=0)
        std += torch.sum(torch.std(data[0].view(now_batch_size, c, h * w), dim=2), dim=0)
    return mean/dataset_size, std/dataset_size
