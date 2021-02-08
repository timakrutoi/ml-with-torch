import matplotlib.pyplot as plt
import argparse


def load_acc(path):
    acc = []

    with open(path, 'r') as in_file:
        for line in in_file:
            acc.append(float(line))

    return acc


def load_loss(path):
    loss = []

    with open(path, 'r') as in_file:
        for line in in_file:
            loss.append(float(line))

    return loss


def plot_all(acc, loss, it=100):
    plt.figure()

    plt.subplot(311)
    for model in acc:
        plt.plot(model)
    plt.ylabel('Accuracy')

    plt.subplot(312)
    for model in loss:
        plt.plot(model)
    plt.ylabel('Loss')

    plt.subplot(313)
    for model in loss:
        plt.plot(model[it:2 * it])
    plt.ylabel('Loss (close look)')

    plt.show()


def plot_pair(acc, loss, it=100):
    plt.figure()

    plt.subplot(311)
    plt.plot(acc)
    plt.plot(it, acc[it], '-ok', color='red', markersize=3)
    plt.plot(2 * it, acc[2 * it], '-ok', color='red', markersize=3)
    plt.ylabel('Accuracy')

    plt.subplot(312)
    plt.plot(loss)
    plt.plot(it, loss[it], '-ok', color='red', markersize=3)
    plt.plot(2 * it, loss[2 * it], '-ok', color='red', markersize=3)
    plt.ylabel('Loss')

    plt.subplot(313)
    plt.plot(loss[it:2 * it + 1])
    plt.plot(0, loss[it], '-ok', color='red', markersize=3)
    plt.plot(it, loss[2 * it], '-ok', color='red', markersize=3)
    plt.ylabel('Loss (close look)')

    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='models numbers selection')
    parser.add_argument('-n', action='store', dest='models_num', default='all')
    parser.add_argument('-d', action='store', dest='dataset', default='test')
    args = parser.parse_args()

    iterations = 50

    names = ['12', '13', '14', '23', '24', '34']
    datasets = {
        'test': 'test_dataset',
        'train': 'train_dataset'
    }

    acc_data = []
    loss_data = []

    try:
        current_dataset = datasets[args.dataset]
    except KeyError:
        print('===>> KeyError: No such dataset')
        print('===>> dataset set as test')
        current_dataset = datasets['test']

    if args.models_num in names:
        acc_data = load_acc('research\\{}\\acc_{}.txt'.format(current_dataset, args.models_num))
        loss_data = load_loss('research\\{}\\loss_{}.txt'.format(current_dataset, args.models_num))

        plot_pair(acc_data, loss_data, iterations)

    elif args.models_num == 'all':
        for i in names:
            acc_data.append(load_acc('research\\{}\\acc_{}.txt'.format(current_dataset, i)))
            loss_data.append(load_loss('research\\{}\\loss_{}.txt'.format(current_dataset, i)))
            plot_all(acc_data, loss_data)

    print('Done!')
