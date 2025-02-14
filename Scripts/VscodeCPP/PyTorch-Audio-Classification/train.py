import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import swanlab

import argparse
from utils import progress_bar #进度条

def create_dataset_csv():
    '''
    创建数据集CSV文件
    '''
    # 数据集根目录
    data_dir = './GTZAN/genres_original'
    data = []
    
    # 遍历所有子目录
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            # 遍历子目录中的所有wav文件
            for audio_file in os.listdir(label_dir):
                if audio_file.endswith('.wav'):
                    audio_path = os.path.join(label_dir, audio_file)
                    data.append([audio_path, label])
    
    # 创建DataFrame并保存为CSV
    df = pd.DataFrame(data, columns=['path', 'label'])
    df.to_csv('audio_dataset.csv', index=False)
    return df

# 自定义数据集类
class AudioDataset(Dataset):
    def __init__(self, df, resize, train_mode=True):
        self.audio_paths = df['path'].values
        # 将标签转换为数值
        self.label_to_idx = {label: idx for idx, label in enumerate(df['label'].unique())}
        self.labels = [self.label_to_idx[label] for label in df['label'].values]
        self.resize = resize
        self.train_mode = train_mode  # 添加训练模式标志
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        # 加载音频文件
        waveform, sample_rate = torchaudio.load(self.audio_paths[idx])
        
        # 将音频转换为梅尔频谱图
        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            hop_length=640,
            n_mels=128
        )
        mel_spectrogram = transform(waveform)

        # 确保数值在合理范围内
        mel_spectrogram = torch.clamp(mel_spectrogram, min=0)
        
        # 转换为3通道图像格式 (为了适配ResNet)
        mel_spectrogram = mel_spectrogram.repeat(3, 1, 1)
        
        # 确保尺寸一致
        resize = torch.nn.AdaptiveAvgPool2d((self.resize, self.resize))
        mel_spectrogram = resize(mel_spectrogram)
        
        return mel_spectrogram, self.labels[idx]

# 修改ResNet模型
class AudioClassifier(nn.Module):
    def __init__(self, num_classes):
        super(AudioClassifier, self).__init__()
        # 加载预训练的ResNet
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        #self.resnet = models.resnet18(pretrained=True)
        # 修改最后的全连接层
        self.resnet.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        return self.resnet(x)

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, start,num_epochs, device):
    
    for epoch in range(start,start+num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            
            running_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss/len(train_loader)
        train_acc = 100.*correct/total

        # 打印进度条，显示当前批次的损失和准确率
        progress_bar(i, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss/(i+1), 100.*correct/total, correct, total))
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx ,(inputs, labels) in enumerate(val_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss/len(val_loader)
        val_acc = 100.*correct/total
        
        current_lr = optimizer.param_groups[0]['lr']
        # 打印进度条，显示当前批次的损失和准确率
        progress_bar(batch_idx, len(val_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (val_loss/(batch_idx+1), 100.*correct/total, correct, total))
        # 记录训练和验证指标
        swanlab.log({
            "train/loss": train_loss,
            "train/acc": train_acc,
            "val/loss": val_loss,
            "val/acc": val_acc,
            "train/epoch": epoch,
            "train/lr": current_lr
        })
        
            
        print(f'Epoch {epoch+1}:')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        print(f'Learning Rate: {current_lr:.6f}')


if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 通过parser设置参数
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--resume', '-r', default=False ,action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--batch_size',default=16)
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--num_epochs',default=20)
    parser.add_argument('--resize',default=224,type=int,help='resize')
    
    args = parser.parse_args()
    #加载进swanlab中
    run = swanlab.init(
        project="PyTorch_Audio_Classification-newest",
        experiment_name ="resnet18",
        config=args
    )
    # 生成或加载数据集CSV文件
    if not os.path.exists('audio_dataset.csv'):
        df = create_dataset_csv()
    else:
        df = pd.read_csv('audio_dataset.csv')

    # 划分训练集和验证集
    train_df = pd.DataFrame()  # train_df 是训练集
    val_df = pd.DataFrame()  # val_df 是测试集
    '''数据处理'''
    for label in df['label'].unique():
        label_df = df[df['label'] == label]  # 获取某一种音乐类型，比如只获取rock
        label_train, label_val = train_test_split(label_df, test_size=0.2, random_state=42)  # 将数据进行拆分 训练集为80% 测试集为20%
        train_df = pd.concat([train_df, label_train])
        val_df = pd.concat([val_df, label_val])

    # 创建数据集和数据加载器
    train_dataset = AudioDataset(train_df, resize=args.resize, train_mode=True)
    val_dataset = AudioDataset(val_df, resize=args.resize, train_mode=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # 创建模型
    num_classes = len(df['label'].unique())  # 根据实际分类数量设置
    print("num_classes", num_classes)
    model = AudioClassifier(num_classes).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # 判断是否需要从 checkpoint 恢复模型
    if args.resume:
        # 加载 checkpoint
        print('==> Resuming from checkpoint..')
        # 断言 checkpoint 目录存在
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        # 加载 checkpoint 文件
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        # 加载模型参数
        model.load_state_dict(checkpoint['model'])
        # 加载最佳准确率
        best_acc = checkpoint['acc']
        # 加载起始 epoch
        start_epoch = checkpoint['epoch']
    else: 
        start_epoch = 0
    # 训练模型
    train_model(model, train_loader, val_loader, criterion, optimizer,start=start_epoch,num_epochs=args.num_epochs,device=device)
    swanlab.finish()
