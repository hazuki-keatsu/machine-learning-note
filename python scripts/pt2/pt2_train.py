import pandas as pd
from tqdm import tqdm
import torch as th
import numpy as np

# 数据加载与预处理
train_date = pd.read_csv('./2024.csv').values

x_train = train_date[:, :-1]
y_train = train_date[:, -1]

print(x_train.shape, y_train.shape)

class TempDataSet(th.utils.data.Dataset):
    def __init__(self, x, y):
        if y is None:
            self.y = y
        else:
            self.y = th.FloatTensor(y)
        self.x = th.FloatTensor(x)

    def __getitem__(self, index):
        if self.y is None:
            return self.x[index]
        else:
            return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

class MyModel(th.nn.Module):
    def __init__(self, input_dim):
        super(MyModel, self).__init__()
        self.layers = th.nn.Sequential(
            th.nn.Linear(input_dim, 64),
            th.nn.ReLU(),
            th.nn.Linear(64, 32),
            th.nn.ReLU(),
            th.nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1)
        return x

# 初始化模型
model = MyModel(input_dim=x_train.shape[1])
criterion = th.nn.MSELoss(reduction='mean')
optimizer = th.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)

# 数据加载
train_dataset = TempDataSet(x_train, y_train)
train_dataloader = th.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)

# 训练循环
for epoch in range(2000):
    model.train()
    train_pbar = tqdm(train_dataloader, position=0, leave=True)
    for x, y in train_pbar:
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

th.save(model.state_dict(), './model.pt')
