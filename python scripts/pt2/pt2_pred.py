import torch as th

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
    
model = MyModel(input_dim=1)

model.load_state_dict(th.load('./model.pt', weights_only=True))
model.eval()

model.eval()
x_input = input('请输入日期：')
x_input = float(x_input)
with th.no_grad():
    x = th.FloatTensor([x_input]).unsqueeze(0)
    pred = model(x)
    pred = pred.item()
    print(f'预测结果：{pred}')