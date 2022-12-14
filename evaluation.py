#!/usr/bin/env python3
import warnings
import torch
import torch.utils
import torch.utils.data
import numpy as np
from data_Prep import Dataset_LOL
import Actions as Actions
import Network
import tqdm
import mcts
import argparse
from ptcolor import deltaE94, rgb2lab

warnings.filterwarnings("ignore")
NUM_ACTIONS = 37
MAX_DEPTH = 10
STOP_ACTION = NUM_ACTIONS - 1
IMAGE_SIZE = 256


def parse_args():
    parser = argparse.ArgumentParser("Compute performace statistics.")
    a = parser.add_argument
    a("base_dir", help="dataset BASE Directory")
    a("weight_file", help="File storing the weights of the CNN")
    a("-s", "--steps", type=int, default=1000, help="Number of MCTS steps")
    a("-e", "--exploration", type=float, default=10, help="Exploration coefficient")
    a("-q", "--initial-q", type=float, default=0.5, help="Value for non-visited nodes")
    a("-b", "--batch-size", type=int, default=30, help="Size of the mini batches")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    a("-d", "--device", default=device, help="Computing device")
    return parser.parse_args()


class EvalState:
    def __init__(self, image, depth=0):
        self.image = image
        self.depth = depth
        self.stopped = False

    def transition(self, action):
        new_image = Actions.select(self.image[None], action)[0]
        new_state = type(self)(new_image, self.depth + 1)
        new_state.stopped = (action == STOP_ACTION)
        return new_state

    def terminal(self):
        return self.depth >= MAX_DEPTH or self.stopped


def play_tree(net, images, device, steps, initial_q, exploration):
    actions = STOP_ACTION + 1
    samples = []

    def transition(states, actions):
        return [s.transition(a) for s, a in zip(states, actions)]

    def evaluation(states):
        t = [s.terminal() for s in states]
        batch = torch.stack([s.image for s in states], 0)
        batch = batch.to(device)
        with torch.no_grad():
            pi, values = net(batch)
            pi = pi.cpu().numpy()
            values = values.squeeze(1).cpu().numpy()
        return t, values, pi

    root_states = [EvalState(im) for im in images]
    trees = mcts.MCTS(root_states, actions, transition, evaluation,
                      exploration=exploration, initial_q=initial_q)
    trees.grow(steps)
    return trees


def mse_error(x, y):
    diff = (x - y).reshape(x.size(0), -1)
    return (diff ** 2).mean(1)


def average_psnr(mses):
    mses = np.maximum(np.array(mses), 1e-6)
    return (-10 * np.log10(mses)).mean()


def eval_closest_node(trees, targets):
    mses = []
    for n in range(trees.roots):
        sub = trees.subtree(n)
        images = torch.stack([s.image for s in trees.x[sub]], 0)
        mse = mse_error(images, targets[n:n + 1]).min()
        mses.append(mse)
    return mses


def eval_most_valuable_node(trees, targets):
    mses = []

    def key(i):
        return trees.R[i] if trees.T[i] else -1

    for n in range(trees.roots):
        sub = trees.subtree(n)
        best = max(sub, key=key)
        image = trees.x[best].image[None]
        mse = mse_error(image, targets[n:n + 1])
        mses.append(mse.item())
    return mses


def evaluation(val_loader, res, args):
    res.eval()
    mses = []
    closest_mses = []
    valuable_mses = []
    l2s = []
    diz = {k: 0 for k in range(NUM_ACTIONS)}
    diz[-1] = 0
    for img, target, name in tqdm.tqdm(val_loader):
        trees = play_tree(res, img, args.device, args.steps, args.initial_q, args.exploration)
        paths, actions, depths = trees.most_visited()
        leaves = paths[np.arange(depths.size), depths - 1]
        enhanced = torch.stack([s.image for s in trees.x[leaves]], 0)
        for i in range(enhanced.shape[0]):
            act = actions[i]
            for ac in act:
                diz[ac] += 1
                if ac == STOP_ACTION:
                    break
            l2s.append(torch.dist(enhanced[i], target[i], 2))
        mse = mse_error(enhanced, target)
        mses.extend(mse.tolist())
        deltae = deltaE94(rgb2lab(enhanced), rgb2lab(target))
        l2s = (torch.stack(l2s, 0)).mean()
        closest_mses.extend(eval_closest_node(trees, target))
        valuable_mses.extend(eval_most_valuable_node(trees, target))
    print(diz)
    print(f"PSNR {average_psnr(mses):.3f}")


def main():
    args = parse_args()
    print('STEPS:', args.steps)
    BASEDIR = args.basedir
    raw_dir = BASEDIR+'TEST/low/'
    exp_dir = BASEDIR+'TEST/high/'

    res = Network.ModifiedResnet(NUM_ACTIONS, 0.0)
    res.to(args.device)
    print("Loading", args.weight_file)
    weights = torch.load(args.weight_file, map_location=args.device)
    res.load_state_dict(weights)

    val_set = Dataset_LOL(raw_dir, exp_dir, size=IMAGE_SIZE, training=False)
    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size=args.batch_size,
                                             drop_last=False,
                                             shuffle=False,
                                             num_workers=1)
    import time
    start = time.time()
    evaluation(val_loader, res, args)
    print('ELAPSED:', time.time() - start)


if __name__ == '__main__':
    import resource

    GB = (2 ** 30)
    mem = 30 * GB
    resource.setrlimit(resource.RLIMIT_DATA, (mem, resource.RLIM_INFINITY))
    main()
