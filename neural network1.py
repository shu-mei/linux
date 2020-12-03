import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter
writer = SummaryWriter('runs/scalar_example')



# 数据预处理：标准化图像数据，使得灰度数据在-1到+1之间
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# 下载Fashion-MNIST训练集数据，并构建训练集数据载入器trainloader,每次从训练集中载入64张图片，每次载入都打乱顺序
trainset = datasets.FashionMNIST('dataset/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# 下载Fashion-MNIST测试集数据，并构建测试集数据载入器trainloader,每次从测试集中载入64张图片，每次载入都打乱顺序
testset = datasets.FashionMNIST('dataset/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)

        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x1))
        x3 = F.relu(self.fc3(x2))
        x4 = F.relu(self.fc4(x3))
        x5 = F.log_softmax(x4, dim=1)

        return x1, x2, x3, x4, x5

def calculate_scatter_matrix(features, labels):
    class_num = torch.max(labels) + 1
    sample_num, feature_dim = features.size()
    u_compare = torch.zeros(feature_dim, class_num)
    u_i = torch.zeros(class_num, feature_dim)
    u = torch.zeros((1, feature_dim))
    N_i = torch.zeros(class_num)
    N = torch.zeros(1)
    for index, i in enumerate(labels):
        u_i[i, :] += features[index, :]
        N_i[i] += 1
        u[0, :] += features[index, :]
        N[0] += 1
    for i in range(class_num):
        if N_i[i] != 0:
            u_i[i, :] = u_i[i, :] / N_i[i]

    u[0, :] = u[0, :] / N[0]

    u_i = u_i.t()
    u = u.t()
    features = features.t()

    s_w = torch.zeros(feature_dim, feature_dim)
    s_b = torch.zeros(feature_dim, feature_dim)

    for index, i in enumerate(labels):
        s_w += torch.mul((u_i[:, i] - features[:, index]), (u_i[:, i] - features[:, index]).t())
        # result = s_w.detach().numpy()
        # if np.isnan(result).any():
        #     np.savetxt('result.txt', result)

        # path_file_name = './action_{}.txt'.format('s_w')
        # if not os.path.exists(path_file_name):
        #     with open(path_file_name, "w") as f:
        #         print(f)
        #
        # with open(path_file_name, "a") as f:
        #     f.write(result)

    for i in range(class_num):
        s_b += N_i[i] * torch.mul((u_i[:, i] - u[:, 0]), (u_i[:, i] - u[:, 0]).t())

        # result1 = s_b.detach().numpy()
        # if np.isnan(result1).any():
        #     np.savetxt('result1.txt', result1)

    return s_w, s_b

def compute_loss(s_w, s_b):

    loss_w = []
    q = 1
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            layer_weight = layer.weight
            loss_w.append(torch.sum(torch.sum(layer_weight.t() ** 2, dim=1) ** (q / 2)) ** (1 / q))

    loss3 = sum(loss_w) / len(loss_w)

    return loss3

def calc_sparsity_solution(param, dim, reg):
    eps = 1e-8
    n = torch.norm(param.t(), p=2, dim=dim)
    scale = torch.max(1 - reg / (n + eps), torch.zeros_like(n))
    scale = scale.repeat(param.shape[0], 1).t()
    return scale

def proximal_operator(lr, regularization):
    for layer in model.modules():

        if isinstance(layer, nn.Linear):
            layer_weight = layer.weight
            reg = regularization * lr
            scale = calc_sparsity_solution(layer_weight, 1, reg)
            layer_weight.data = torch.mul(scale, layer_weight.t()).t()
            # print(layer_weight)

def reg_anneal(lossp, regularization_factor, annealing_factor, annealing_t1, annealing_t2):
    """
    Anneal the regularization factor
    :param lossp:
    :param regularization_factor:
    :param annealing_factor:
    :param annealing_t1:
    :param annealing_t2:
    :return:
    """
    if annealing_factor == 0:
        regularization = regularization_factor
    else:
        if annealing_t2 < lossp <= annealing_t1:
            regularization = regularization_factor / annealing_factor
        elif lossp <= annealing_t2:
            regularization = regularization_factor / annealing_factor ** 2
        else:
            regularization = regularization_factor
    return regularization

# 对上面定义的Classifier类进行实例化
model = Classifier()

# 定义损失函数为负对数损失函数
# criterion = nn.NLLLoss()
criterion = nn.CrossEntropyLoss()
criterion1 = nn.MSELoss()

# 优化方法为Adam梯度下降方法，学习率为0.003
# optimizer = optim.Adam(model.parameters(), lr=0.003)
optimizer = optim.SGD(model.parameters(), lr=0.001)

# 对训练集的全部数据学习15遍，这个数字越大，训练时间越长
epochs = 200

# 将每次训练的训练误差和测试误差存储在这两个列表里，后面绘制误差变化折线图用
train_losses, test_losses = [], []

print('开始训练')
for e in range(epochs):
    running_loss = 0

    # 对训练集中的所有图片都过一遍
    batch = 0
    for images, labels in trainloader:
        batch = batch+1
        # 将优化器中的求导结果都设为0，否则会在每次反向传播之后叠加之前的
        optimizer.zero_grad()

        # 对64张图片进行推断，计算损失函数，反向传播优化权重，将损失求和
        log_ps = model(images)
        b, c, w, h = images.size()
        images = images.reshape(b, c*w*h)
        s_w = []
        s_b = []
        for i in range(5):
            if i==0:
                w, b = calculate_scatter_matrix(images, labels)
                s_w.append(w)
                s_b.append(b)
            else:
                w, b = calculate_scatter_matrix(log_ps[i-1], labels)
                s_w.append(w)
                s_b.append(b)

        loss1 = 0
        loss2 = 0
        i = 0
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                layer_weight = layer.weight
                # layer_weight = layer_weight.permute(1, 0)
                layer_weight = torch.mm(torch.mm(layer_weight, s_w[i]), layer_weight.t())
                loss1 += torch.abs(torch.trace(layer_weight))
                # writer.add_scalar('loss1', loss1.item(), i)
                i = i + 1
        j = 0
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                layer_weight = layer.weight
                # layer_weight = layer_weight.permute(1, 0)
                layer_weight = torch.mm(torch.mm(layer_weight, s_b[j]), layer_weight.t())

                n, _ = layer_weight.size()

                loss2 += criterion1(layer_weight, torch.eye(n))
                # loss2 += - criterion1(layer_weight, torch.eye(n))
                # writer.add_scalar('loss2', loss2.item(), j)
                j = j + 1

        loss = criterion(log_ps[4], labels)

        loss = loss + 0.001 * loss1 + 0.01 * loss2

        loss3 = compute_loss(s_w, s_b)

        loss.backward()

        optimizer.step()

        regularization_factor = 0.0004
        annealing_factor = 2.0
        annealing_t1 = 10
        annealing_t2 = 5
        lr = 0.1

        reg = reg_anneal(loss3, regularization_factor, annealing_factor,
                         annealing_t1, annealing_t2)
        proximal_operator(lr, reg)
        # print('loss:', loss)
        # print('loss1:', loss1)
        # print('loss2:', loss2)

        # loss = loss + 0.001*loss1 + 0.01*loss2

        # loss = loss + 0.0001 * loss1
        # loss = loss + 0.01*loss2

        # optimizer.step()
        running_loss += loss.item()
        # print('running_loss:', running_loss)

    # 每次学完一遍数据集，都进行以下测试操作
    else:
        test_loss = 0
        accuracy = 0
        # 测试的时候不需要开自动求导和反向传播
        with torch.no_grad():
            # 关闭Dropout
            model.eval()

            # 对测试集中的所有图片都过一遍
            for images, labels in testloader:
                # 对传入的测试集图片进行正向推断、计算损失，accuracy为测试集一万张图片中模型预测正确率
                log_ps = model(images)
                test_loss += criterion(log_ps[4], labels)
                ps = torch.exp(log_ps[4])
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)

                # 等号右边为每一批64张测试图片中预测正确的占比
                accuracy += torch.mean(equals.type(torch.FloatTensor))
        # 恢复Dropout
        model.train()
        # 将训练误差和测试误差存在两个列表里，后面绘制误差变化折线图用
        train_losses.append(running_loss / len(trainloader))
        test_losses.append(test_loss / len(testloader))

        print("训练集学习次数: {}/{}.. ".format(e + 1, epochs),
              "训练误差: {:.3f}.. ".format(running_loss / len(trainloader)),
              "测试误差: {:.3f}.. ".format(test_loss / len(testloader)),
              "模型分类准确率: {:.3f}".format(accuracy / len(testloader)))