import matplotlib.pyplot as plt
import matplotlib.style as style

from dataset import *
from model import *

def train(model, train_dataloader, criterion, scheduler, optimizer, epochs, batch_size, device):
    train_loss_list = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs[:, :4]
            # labels와 차원 맞추기
            # outputs = model(inputs).squeeze()
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            # Batch 내 데이터 loss 평균 * Batch 내 데이터 개수
            running_loss += loss.item()
        
        scheduler.step()

        # 전체 데이터에 대한 loss 총합 / 전체 데이터 개수
        epoch_loss = running_loss / (len(train_dataloader.dataset) / batch_size)
        train_loss_list.append(epoch_loss)

        print('Epoch [{}/{}], train_loss: {:.4f}'.format(epoch + 1, epochs, epoch_loss))

    
    return model, train_loss_list


def show_loss(train_loss_list, val_loss_list, title='MSE Loss'):
    train_x = range(len(train_loss_list))
    train_y = train_loss_list
    val_x = range(len(val_loss_list))
    val_y = val_loss_list

    # 첫 번째 축을 생성하고 train loss를 그립니다
    fig, ax1 = plt.subplots()
    ax1.plot(train_x, train_y, label='train loss')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('train loss')
    ax1.tick_params(axis='y')

    # 두 번째 축을 생성하고 validation loss를 그립니다
    ax2 = ax1.twinx()
    ax2.plot(val_x, val_y, label='validation loss', color='darkorange')
    ax2.set_ylabel('validation loss')
    ax2.tick_params(axis='y')

    # 범례를 표시합니다
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='best')

    plt.title(title)
    plt.savefig('./train loss/' + title + '.png')
    # plt.show()