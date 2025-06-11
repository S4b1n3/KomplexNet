import argparse
parser = argparse.ArgumentParser(description="Complex-Valued neural networks")

parser.add_argument('--frame_index', type=int, default=0)

parser.add_argument('--model', type=str, default='small_complex', choices=['big_complex', 'big_real', 'small_complex',
                                                                     'small_real', 'small_vit'], help='Model type')
parser.add_argument('--use_fb', type=eval, default=False, choices=[True, False])
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--n_channels', type=int, default=8)
parser.add_argument('--biases', type=eval, default=True, choices=[True, False])
parser.add_argument('--timesteps', type=int, default=15)
parser.add_argument('--dataset', type=str, default='multi_mnist', choices=['multi_mnist', 'svhn', 'multi_mnist_cifar', 'multi_mnist_svhn', 'multi_mnist_cifar2', 'tetrominoes', 'norb'])
parser.add_argument('--add_noise', type=eval, default=False, choices=[True, False])
parser.add_argument('--num_objects', type=int, default=2)

parser.add_argument('--kuramoto_mode', type=str, default='endtoend', choices=['channels', 'input', 'endtoend'])
parser.add_argument('--kernel', type=str, default='gaussian', choices=['learnt', 'gaussian', 'random'])
parser.add_argument('--gabors', type=eval, default=True, choices=[True, False])
parser.add_argument('--bg_kernel', type=eval, default=False, choices=[True, False])
parser.add_argument('--random', type=eval, default=False, choices=[True, False])
parser.add_argument('--phase_mode', type=str, default='kuramoto', choices=['random', 'ideal', 'kuramoto'])

parser.add_argument('--epsilon', type=float, default=0.202)
parser.add_argument('--epsilon_l2', type=float, default=0)
parser.add_argument('--lr_kuramoto', type=float, default=0.006)
parser.add_argument('--lr_kuramoto_l2', type=float, default=0.006)
parser.add_argument('--lr_kuramoto_l3', type=float, default=0.006)
parser.add_argument('--lr_kuramoto_l4', type=float, default=0.006)
parser.add_argument('--mean_r', type=float, default=0)
parser.add_argument('--std_r', type=float, default=3)
parser.add_argument('--std_l2', type=float, default=3)
parser.add_argument('--coef', type=float, default=1)
parser.add_argument('--h', type=int, default=13)
parser.add_argument('--w', type=int, default=13)
parser.add_argument('--k', type=int, default=13)
parser.add_argument('--k_l2', type=int, default=11)
parser.add_argument('--k_l3', type=int, default=11)
parser.add_argument('--loss_coef', type=float, default=1)

parser.add_argument('--save', type=str, default='./results/')
parser.add_argument('--filename', type=str, default='complex_model_endtoend_learnablefb_8channels')
parser.add_argument('--in_repo', type=str, default='object_0')
parser.add_argument('--out_repo', type=str, default='2iqtgodw')
parser.add_argument('--saved', type=eval, default=False, choices=[True, False])
parser.add_argument('--resume', type=eval, default=False, choices=[True, False])

parser.add_argument('--data_aug', type=eval, default=False, choices=[True, False])
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--test_batch_size', type=int, default=256)
parser.add_argument('--seed', type=int, default=1)

parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-parallel', '--parallel', default= False, action='store_true',
                    help='Wanna parallelize the training')
parser.add_argument('--gpu', type=str, default="5")
parser.add_argument('--monitor', type=eval, default=False, choices=[True, False])
