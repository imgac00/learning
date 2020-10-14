from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# Set random seed for reproducibility设置随机种子进行重现性
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results如果你想要新的结果，就使用
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

"""dataroot -数据集文件夹根目录的路径。 我们将在下一节中进一步讨论数据集
worker -使用 DataLoader 加载数据的工作线程数
batch_size -训练中使用的批次大小。 DCGAN 纸使用的批处理大小为 128
image_size -用于训练的图像的空间大小。 此实现默认为 64x64。 如果需要其他尺寸，则必须更改 D 和 G 的结构。 有关更多详细信息，请参见此处的。
nc -输入图像中的颜色通道数。 对于彩色图像，这是 3
nz -潜矢量的长度
ngf -与通过生成器传送的特征图的深度有关
ndf -设置通过鉴别器传播的特征图的深度
num_epochs -要运行的训练时期数。 训练更长的时间可能会导致更好的结果，但也会花费更长的时间
lr -训练的学习率。 如 DCGAN 文件中所述，此数字应为 0.0002
beta1 -Adam 优化器的 beta1 超参数。 如论文所述，该数字应为 0.5
ngpu -可用的 GPU 数量。 如果为 0，代码将在 CPU 模式下运行。 如果此数字大于 0，它将在该数量的 GPU 上运行"""


# Root directory for dataset
dataroot = "./data/celeba"

# Number of workers for dataloader  dataloader的工作线程数
workers = 0

# Batch size during training培训期间批量大小
batch_size = 128

# Spatial size of training images. All images will be resized to this训练图像的空间大小。 所有的图像将被调整到这个大小
#   size using a transformer.使用变压器的尺寸
image_size = 64

# 训练图像中的通道数。 对于彩色图像，这是3 Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)z潜矢量的大小(即。 gen输入的大小)
nz = 100

# Size of feature maps in generator# 生成器中特征映射的大小
ngf = 64

# Size of feature maps in discriminator鉴别器中特征映射的大小
ndf = 64

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers优化器的学习速率
lr = 0.0002

# Beta1 hyperparam for Adam optimizers用于亚当优化器
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1


# We can use an image folder dataset the way we have it setup.我们可以使用图像文件夹数据集的方式，我们有它的设置
# Create the dataset创建数据集 Compose构成 Resize调整大小 CenterCrop
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size), #中间区域进行裁剪
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader创建数据处理程序
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Decide which device we want to run on决定我们要运行哪个设备
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Plot some training images 绘制一些训练图像
real_batch = next(iter(dataloader))
plt.figure(figsize=(8, 8))# 图，表; 几何图形
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
# plt.show()

# custom weights initialization called on netG and netD调用自定义权重初始化
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution卷积 stride步 padding
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False), #偏向
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# Create the generator
netG = Generator(ngpu).to(device)

# Handle multi-gpu if desired如果需要，多GPU句柄
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights应用weights_init函数随机初始化所有权重
#  to mean=0, stdev=0.2.
netG.apply(weights_init)

# Print the model
print(netG)

# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Print the model
print(netD)

# Initialize BCELoss function初始化BCELoss函数
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize创建一批潜在向量，我们将使用这些向量来可视化
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training在培训期间建立真假标签的惯例
real_label = 1
fake_label = 0

# Setup Adam optimizers for both G and D为G和D设置亚当优化器
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Training Loop

# Lists to keep track of progress列出跟踪进度
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))更新D网：最大化
        ###########################
        ## Train with all-real batch以全真实批
        netD.zero_grad()
        # Format batch格式批处理
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, out=None, dtype=torch.float32, device=device)
        # Forward pass real batch through D向前通过真实批次通过D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch计算所有实际批次的损失
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass计算后向传球中D的梯度
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors生成一批潜在向量
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G 用G生成假图像批处理
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D用D对所有假批次进行分类
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch计算D在全饼批次上的损失
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch计算此批的梯度
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches 添加所有真实和所有蛋糕批次的梯度
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost假标签是真实的发电机成本
        # Since we just updated D, perform another forward pass of all-fake batch through D
        #由于我们刚刚更新了D，所以通过D执行另一个全假批的转发传递
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output根据这个输出计算G的损失
        errG = criterion(output, label)
        # Calculate gradients for GCalculate gradients for G计算G的梯度
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats输出训练统计数据
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later为以后的绘图节省损失
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        # 通过在fixed_noise上保存G的输出来检查生成器是如何做的
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()


#%%capture 展现;
fig = plt.figure(figsize=(8, 8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())
