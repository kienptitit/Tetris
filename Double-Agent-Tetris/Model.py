import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import deque


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(nn.Linear(4, 64), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Linear(64, 1))

        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

    def save(self, file_name='model.pth'):
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.gamma = gamma
        self.lr = lr
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done, long_term=False):
        """
        state : [BATCH_SIZE , 20*17] (Single Mode),[BATCH_SIZE,20*34] (Two Mode) (torch.tensor())
        action : ([action1],[action2],....[action_BATCHSIZE])
        reward : (reward1,reward2,...reward_BATCHSIZE)
        next_state : same_as state
        """
        state, next_state = torch.tensor(state).type(torch.float).to('cuda'), torch.tensor(next_state).type(
            torch.float).to('cuda')
        action = torch.tensor(action).to('cuda')
        reward = torch.tensor(reward).to('cuda')

        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            action = action.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done,)

        pred = self.model(state)  # [Batch_size,8]

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = Q_new + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
        return loss


if __name__ == '__main__':
    import numpy as np

    check = 0
    for i in range(4):
        for j in range(4):
            if j > 2:
                check = 1
    print(check)
    # a = np.array([[1, 2, 3], [2, 3, 4]])
    # b = np.array([[1, 2, 3], [2, 3, 4]])
    #
    # s = set()
    # s.add(tuple(a))
    # s.add(tuple(b))
    # print(s)
    # a = torch.tensor([1.0, 3.0, 2.0]).type(torch.long)
    # print(a.dtype)
    # m = deque(maxlen=100)
    # m.append((torch.tensor(1), torch.tensor(2)))
    # m.append((torch.tensor(3), torch.tensor(4)))
    # x1, x2 = zip(*m)
    # print(x1)
    # print(x2)
# inp = torch.randn(20, 20 * 17)
# m = Model()
# print(m(inp).shape)
