import datetime


# 开始训练
def print_start_train():
    print("Start Training...")
    now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("==========" * 8 + "%s" % now_time)


# 批次损失
def print_batch_loss(step, loss):
    """
    :param step: 当前批次
    :param loss: 当前批次损失
    """
    print("[step = %d] loss: %.3f, " % (step, loss))


# epoch 损失
def print_epoch_loss(evaluation, evaluation_name):
    """
    :param evaluation: (epoch, loss_sum, evaluation, val_loss_sum, val_evaluation)
    :param evaluation_name: "accuracy" or "f1"
    """
    print(("\nEPOCH = %d, loss = %.3f, " + evaluation_name +
           " = %.3f, val_loss = %.3f, " + "val_" + evaluation_name + " = %.3f")
          % evaluation)
    now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n" + "==========" * 8 + "%s" % now_time)


# 测试损失
def print_test_loss(evaluation, evaluation_name):
    """
    :param evaluation: (loss_sum, evaluation)
    :param evaluation_name: "accuracy" or "f1"
    """
    print(("\nTEST: loss = %.3f, " + evaluation_name + " = %.3f")
          % evaluation)
    now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n" + "==========" * 8 + "%s" % now_time)


# grid search
def print_grid_search_result(best_params, best_acc):
    """
    :param best_params: dict
    :param best_acc: float
    """
    print("Grid Search Result:")
    print("Best params: ", best_params)
    print("Best acc: ", best_acc)
    now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n" + "==========" * 8 + "%s" % now_time)



# 结束训练
def print_end_train():
    print('Finished Training...')


if __name__ == '__main__':
    print_start_train()
    print_batch_loss(1, 2, "accuracy", 3)
    print_batch_loss(2, 3, "accuracy", 3)
    print_epoch_loss((1, 2, 3, 4, 5), "accuracy")
    print_end_train()
