import matplotlib.pyplot as plt

if __name__ == '__main__':

    acc_data = []
    loss_data = []

    with open('research_acc_data.txt', 'r') as in_file:
        for line in in_file:
            acc_data.append(float(line))

    with open('research_loss_data.txt', 'r') as in_file:
        for line in in_file:
            loss_data.append(float(line))

    plt.figure()

    plt.subplot(211)
    plt.plot(acc_data)
    plt.ylabel('Accuracy')

    plt.subplot(212)
    plt.plot(loss_data)
    plt.ylabel('Loss')

    plt.show()

    print('Done!')
