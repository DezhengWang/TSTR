import os
import torch
import time


def adjust_learning_rate(optimizer, epoch, args, type="type2"):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if type=='type1':
        lr_adjust = {epoch: args.lr * (0.5 ** ((epoch-1) // 1))}
    elif type=='type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        args.lr = lr


class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean


def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def save_model(net, path):
    with open(os.path.join(path, "net.pt"), 'wb') as f:
        torch.save(net, f)


def mkdir(model, dataset, pred_len, augmentation="None", index=0):
    path = os.path.join("Result", pred_len, model, dataset, augmentation, str(index))
    if not os.path.exists("Result"):
        os.mkdir("Result")
    if not os.path.exists(os.path.join("Result", pred_len)):
        os.mkdir(os.path.join("Result", pred_len))
    if not os.path.exists(os.path.join("Result", pred_len, model)):
        os.mkdir(os.path.join("Result", pred_len, model))
    if not os.path.exists(os.path.join("Result", pred_len, model, dataset)):
        os.mkdir(os.path.join("Result", pred_len, model, dataset))
    if not os.path.exists(os.path.join("Result", pred_len, model, dataset, augmentation)):
        os.mkdir(os.path.join("Result", pred_len, model, dataset, augmentation))
    if not os.path.exists(path):
        os.mkdir(path)
        print(f"[INFO] "+time.strftime("%Y-%m-%d %H:%M:%S",time.gmtime())+" The file folder was created: {}".format(path))
    return path
