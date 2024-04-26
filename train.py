import matplotlib.pyplot as plt
import matplotlib.style as style

from dataset import *
from model import *

def train(model, train_dataloader, criterion, scheduler, optimizer, epochs, batch_size, device, val_dataloader=None, patience=10, is_val=True, lambda1=0):
    patience = 0
    train_loss_list = []
    val_loss_list = []
    min_val_loss = float('inf')

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

            # L1 Regularization
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = loss + lambda1 * l1_norm

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            # Batch 내 데이터 loss 평균 * Batch 내 데이터 개수
            running_loss += loss.item()
        
        scheduler.step()

        # 전체 데이터에 대한 loss 총합 / 전체 데이터 개수
        epoch_loss = running_loss / (len(train_dataloader.dataset) / batch_size)
        train_loss_list.append(epoch_loss)

        if is_val:
            model.eval()
            val_running_loss = 0.0

            for val_inputs, val_labels in val_dataloader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_inputs = val_inputs[:, :4]
                # val_outputs = model(val_inputs).squeeze()
                val_outputs = model(val_inputs)

                val_loss = criterion(val_outputs, val_labels)

                val_running_loss += val_loss.item()

            val_epoch_loss = val_running_loss / (len(val_dataloader.dataset) / batch_size)
            val_loss_list.append(val_epoch_loss)

            print('Epoch [{}/{}], train_loss: {:.4f}, val_loss: {:.4f}'.format(epoch + 1, epochs, epoch_loss, val_epoch_loss))

            # Early stopping
            if val_epoch_loss < min_val_loss:
                min_val_loss = val_epoch_loss
                patience = 0
            else:
                if patience > 10:
                    break
                patience += 1
        else:
            print('Epoch [{}/{}], train_loss: {:.4f}'.format(epoch + 1, epochs, epoch_loss))

    
    return model, train_loss_list, val_loss_list


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