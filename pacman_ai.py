import numpy as np                # Xử lý mảng số liệu
import gymnasium as gym           # Môi trường RL (hoặc gym)
import torch                      # PyTorch cho mạng neural
import torch.nn as nn             # Xây dựng mạng neural
import torch.optim as optim       # Tối ưu hóa mạng
from collections import deque     # Replay buffer (bộ nhớ kinh nghiệm)
import random                     # Lấy mẫu ngẫu nhiên từ buffer

# 1. Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done
    def __len__(self):
        return len(self.buffer)

# 2. Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        # 2 hidden layers
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
# 3. DQN Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.95, epsilon=1.0, epsilon_min=0.00005, epsilon_decay=0.99995):
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(100000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.action_dim = action_dim

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state)
        return q_values.argmax().item()

    def train(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
        state, action, reward, next_state, done = self.memory.sample(batch_size)
        state = torch.FloatTensor(state)
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done)

        q_values = self.q_net(state).gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_net(next_state).max(1)[0]
        expected_q = reward + self.gamma * next_q_values * (1 - done)

        loss = nn.MSELoss()(q_values, expected_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    def save(self, path):
        torch.save(self.q_net.state_dict(),path)
    def load(self, path):
        self.q_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.q_net.state_dict())

def check_position(centerx, centery, level):
    turns = [False, False, False, False]  # [Right, Left, Up, Down]
    num1 = (690 - 50) // 32
    num2 = 600 // 30
    num3 = 10

    def can_move(row, col):
        if 0 <= row < len(level) and 0 <= col < len(level[row]):
            return level[row][col] < 3
        return False

    row = centery // num1
    col = centerx // num2

    # Luôn cho phép đi qua đường hầm
    if col >= 29:
        turns[0] = True
        turns[1] = True
        return turns

    # Kiểm tra 4 hướng cơ bản
    if can_move(row, (centerx + num3) // num2):  # RIGHT
        turns[0] = True
    if can_move(row, (centerx - num3) // num2):  # LEFT
        turns[1] = True
    if can_move((centery - num3) // num1, col):  # UP
        turns[2] = True
    if can_move((centery + num3) // num1, col):  # DOWN
        turns[3] = True

    return turns


def get_state(player_x, player_y, ghosts, level, powerup, direction=0, lives=1):
    state = [player_x/600, player_y/690]
    for g in ghosts:
        state.extend([g.x_pos/600, g.y_pos/690])
        state.append(int(g.dead))
        state.append(int(g.in_box))
        state.append(g.direction/3)  # Thêm hướng ghost
        dist = ((player_x - g.x_pos)**2 + (player_y - g.y_pos)**2)**0.5 / 800
        state.append(dist)
    state.append(int(powerup))
    state.append(direction/3)

    center_x = player_x + 15
    center_y = player_y + 15
    turns_allowed = check_position(center_x, center_y, level)
    state.extend([int(t) for t in turns_allowed])

    # Dot gần nhất
    min_dist = float('inf')
    nearest_dot = (player_x, player_y)
    dot_count = 0
    power_min_dist = float('inf')
    nearest_power_dot = (player_x, player_y)
    for i in range(len(level)):
        for j in range(len(level[i])):
            if level[i][j] == 1 or level[i][j] == 2:
                dot_count += 1
                dot_x = j * (600 // 30) + (600 // 60)
                dot_y = i * ((690-50)//32) + ((690-50)//64)
                dist = ((player_x - dot_x)**2 + (player_y - dot_y)**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    nearest_dot = (dot_x, dot_y)
            if level[i][j] == 2:  # power dot
                dot_x = j * (600 // 30) + (600 // 60)
                dot_y = i * ((690-50)//32) + ((690-50)//64)
                dist = ((player_x - dot_x)**2 + (player_y - dot_y)**2)**0.5
                if dist < power_min_dist:
                    power_min_dist = dist
                    nearest_power_dot = (dot_x, dot_y)
    state.extend([nearest_dot[0]/600, nearest_dot[1]/690])
    state.extend([nearest_power_dot[0]/600, nearest_power_dot[1]/690])
    state.append(dot_count / 240)
    state.append(lives/3)

    return np.array(state, dtype=np.float32)

    
    