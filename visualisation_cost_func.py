import matplotlib.pyplot as plt

if __name__ == '__main__':
    trained_point_1 = (100, 0.3992, 207.11342453956604)
    trained_point_2 = (200, 0.4178, 186.5473130941391)
    '''
    100 - 0.3992 - 207.11342453956604
    200 - 0.4178 - 186.5473130941391
    '''

    acc_data = []
    loss_data = []

    with open('research\\research_acc_data.txt', 'r') as in_file:
        for line in in_file:
            acc_data.append(float(line))

    with open('research\\research_loss_data.txt', 'r') as in_file:
        for line in in_file:
            loss_data.append(float(line))

    plt.figure()

    plt.subplot(211)
    plt.plot(acc_data)
    plt.plot(trained_point_1[0], trained_point_1[1], '-ok', color='red', markersize=3)
    plt.plot(trained_point_2[0], trained_point_2[1], '-ok', color='red', markersize=3)
    plt.ylabel('Accuracy')

    plt.subplot(212)
    plt.plot(loss_data)
    plt.plot(trained_point_1[0], trained_point_1[2], '-ok', color='red', markersize=3)
    plt.plot(trained_point_2[0], trained_point_2[2], '-ok', color='red', markersize=3)
    plt.ylabel('Loss')

    plt.show()

    print('Done!')
