from ml_torch import *
import torch.nn as nn


def find_steps(start, finish, iterations=100):
    return (finish - start) / float(iterations)


def set_weights(conv1, conv2, lc):
    model = net()
    model.conv1.weight.data = conv1
    model.conv2.weight.data = conv2
    model.lc.weight.data = lc

    # print(model)
    return model


def net_test(model):
    # Testing
    total_cnt = 0
    correct_cnt = 0
    test_loss = 0
    acc = 0

    for batch_idx, (x, target) in enumerate(test_loader):  # reading test data
        y = model(x)

        loss = criterion(y, target)

        test_loss += loss.item()
        _, predict = y.max(1)
        total_cnt += target.size(0)
        correct_cnt += predict.eq(target).sum().item()
        acc = (correct_cnt * 1.) / total_cnt

    return acc, test_loss


if __name__ == '__main__':

    batch_size = 100
    num_iterations = 100

    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    test_set = CIFAR10(root='./data', train=False, download=True, transform=trans)

    criterion = nn.CrossEntropyLoss()

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False)

    model1 = net()
    model1.load_state_dict(torch.load('calculated models\\model(try1).pt'))
    model1.eval()

    model2 = net()
    model2.load_state_dict(torch.load('calculated models\\model(try2).pt'))
    model2.eval()

    conv1_model1 = model1.conv1.weight
    conv1_model2 = model2.conv1.weight
    conv1_steps = find_steps(conv1_model1, conv1_model2, num_iterations)

    print('conv1 steps have been found!')

    conv2_model1 = model1.conv2.weight
    conv2_model2 = model2.conv2.weight
    conv2_steps = find_steps(conv2_model1, conv2_model2, num_iterations)

    print('conv2 steps have been found!')

    lc_model1 = model1.lc.weight
    lc_model2 = model2.lc.weight
    lc_steps = find_steps(lc_model1, lc_model2, num_iterations)

    print('lc steps have been found!')
    print('All steps have been found!')

    result_list = [(0, 0)]
    test_conv1 = conv1_model1 - (conv1_steps * num_iterations)
    test_conv2 = conv2_model1 - (conv2_steps * num_iterations)
    test_lc = lc_model1 - (lc_steps * num_iterations)

    it = tqdm(range(3 * num_iterations), ncols=140)

    for i in enumerate(it):
        # applying changes
        test_model = set_weights(test_conv1, test_conv2, test_lc)

        # testing model with new weights and writing results
        result_list.append(net_test(test_model))

        # changing weights
        test_conv1 = test_conv1 + conv1_steps
        test_conv2 = test_conv2 + conv2_steps
        test_lc = test_lc + lc_steps

    it.close()

    with open('research\\research_right_acc_data.txt', 'w') as out_file:
        for i in result_list:
            out_file.write('{}\n'.format(i[0]))

    with open('research\\research_right_loss_data.txt', 'w') as out_file:
        for i in result_list:
            out_file.write('{}\n'.format(i[1]))

    print('Done!')
