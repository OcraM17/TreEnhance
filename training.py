#!/usr/bin/env python3
import torch
import torch.utils
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import collections
import numpy as np
import random
from data_Prep import LoadDataset, Dataset_LOL
import Actions as Actions
import warnings
import Network
import os
from matplotlib import pyplot as plt
from metrics import PSNR
import tqdm
import mcts
import argparse
import ptcolor

# !!! Constants to command-line arguments
MAX_DEPTH = 10
STOP_ACTION = 36


def parse_args():
    parser = argparse.ArgumentParser("TreEnhance Hyperparams")
    a = parser.add_argument
    a("basedir", help="BASE DIRECTORY")
    a("expname",help="Name of the run")
    a("dropout", type=float, default=0.6, help="Dropout")
    a("num_images", type=int, default=100, help="number of Images")
    a("num_steps", type=int, default=1000, help="number of steps")
    a("val_images", type=int, default=100, help="number of val images")
    a("lr", type=float, default=0.001, help="learning rate")
    a("size", type=int, default=256, help="image size")
    a("num_gen", type=float, default=256, help="number of generation")
    a("bs", type=int, default=256, help="batch size")
    a("lambd", type=int, default=20, help="lambda in the loss function")
    a("loops", type=int, default=5, help="number of optimization loops")
    return parser.parse_args()

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.zeros_(m.weight)
        torch.nn.init.zeros_(m.bias)


def add_plot(x, y, writer, step):
    plt.scatter(x, y, edgecolors='b')
    plt.xticks(np.arange(0, 1, 0.1))
    plt.yticks(np.arange(0, 1, 0.1))
    plt.xlabel('z')
    plt.ylabel('y')
    plt.title('outcome plot')
    plt.grid(True)
    writer.add_figure('Fig1', plt.gcf(), step)


TrainingSample = collections.namedtuple("TrainingSample", ["image", "return_", "probabilities"])


def compute_error(x, y):
    labx = ptcolor.rgb2lab(x.unsqueeze(0))
    laby = ptcolor.rgb2lab(y.unsqueeze(0))
    de = ptcolor.deltaE94(labx, laby)
    return de


def train(samples, res, optimizer, step, device, writer,
          train_loss_history, train_L1_history, train_L2_history, args, lambd=10):
    img = [s.image.unsqueeze(0) for s in samples]
    prob = [s.probabilities for s in samples]
    win = [s.return_ for s in samples]
    DS = LoadDataset(img, torch.tensor(prob), win)
    L = torch.utils.data.DataLoader(DS, batch_size=64, drop_last=False,
                                    shuffle=True, num_workers=0)
    res.train()
    loops = args.loops
    for loop in tqdm.tqdm(range(loops)):
        z_x, v_y = [], []
        for img_prob in L:
            outcome = img_prob[2].to(device)
            optimizer.zero_grad()
            pred, v = res(img_prob[0][:, 0, :, :, :].to(device))
            z_x += outcome.unsqueeze(1)
            v_y += v
            l1 = lambd * ((outcome.unsqueeze(1) - v) ** 2)
            l2 = -(((torch.tensor(img_prob[1]).to(device) *
                     torch.log(torch.clamp(pred, min=1e-8))).sum(1)))
            loss = ((l1 + l2.unsqueeze(1)).mean())
            train_loss_history.append(loss.item())
            train_L1_history.append(l1.mean().item())
            train_L2_history.append(l2.mean().item())
            loss.backward()
            optimizer.step()
            step += 1
            if step % 10 == 0:
                mean_loss = (sum(train_loss_history) /
                             max(1, len(train_loss_history)))
                mean_L1 = sum(train_L1_history) / max(1, len(train_L1_history))
                mean_L2 = sum(train_L2_history) / max(1, len(train_L2_history))
                writer.add_scalar('Loss', mean_loss, step)
                writer.add_scalar('L1', mean_L1, step)
                writer.add_scalar('L2', mean_L2, step)
                tqdm.tqdm.write(f"{step} {mean_L1} + {mean_L2} = {mean_loss}")
    z_x = torch.cat(z_x, dim=0)
    v_y = torch.cat(v_y, dim=0)
    add_plot(z_x.cpu().detach().numpy(), v_y.cpu().detach().numpy(),
             writer, step)
    writer.add_scalar('Average return', z_x.mean().item(), step)
    return res, step


class TrainingState:
    def __init__(self, image, target, depth=0):
        self.image = image
        self.target = target
        self.depth = depth
        self.stopped = False

    def transition(self, action):
        new_image = Actions.select(self.image[None], action)[0]
        new_state = type(self)(new_image, self.target, self.depth + 1)
        new_state.stopped = (action == STOP_ACTION)
        return new_state

    def terminal(self):
        return self.depth >= MAX_DEPTH or self.stopped

    def compute_return(self):
        if self.depth >= MAX_DEPTH:
            return 0.0
        elif self.stopped:
            d = torch.dist(self.image, self.target, 2)
            return torch.exp(-0.05 * d).item()
        else:
            raise ValueError("This state has not return!")


def play_tree(net, images, targets, device, steps):
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
        if np.all([v.depth == 0 for v in states]):
            eps = 0.25
            pi = (1 - eps) * pi + eps * np.random.dirichlet([0.03 for i in range(STOP_ACTION + 1)],
                                                            pi.shape[0])
        r = [(s.compute_return() if s.terminal() else v.item())
             for (v, s) in zip(values, states)]
        return t, r, pi

    root_states = [TrainingState(im, tgt) for im, tgt in zip(images, targets)]
    trees = mcts.MCTS(root_states, actions, transition, evaluation, exploration=8, initial_q=1.0)
    states = []
    probs = []
    samples = []
    while not np.all(trees.T[:trees.roots]):
        trees.grow(steps)
        states.append(trees.x[:trees.roots])
        tau = 1.0
        numerator = trees.N[:trees.roots, :] ** (1 / tau)
        denominator = np.maximum(1, numerator.sum(1, keepdims=True))
        probs.append(numerator / denominator)
        actions = trees.sample_path()[1]
        trees.descend_tree(actions[:, 0])
    errors = []
    psnrs = []
    for r in range(trees.roots):
        z = trees.R[r]
        for s, p in zip(states, probs):
            if s[r].terminal():
                errors.append(torch.dist(s[r].image, s[r].target, 2).item())
                psnrs.append(PSNR(s[r].image, s[r].target).item())
                break
            samples.append(TrainingSample(s[r].image, z, p[r, :]))
    return samples, errors, psnrs


def generation(res, loader, steps, device):
    samples = []
    errors = []
    psnrs = []
    res.eval()
    for images, targets in tqdm.tqdm(loader):
        s, e, p = play_tree(res, images, targets, device, steps)
        samples.extend(s)
        errors.extend(e)
        psnrs.extend(p)
    return samples, np.mean(errors), np.mean(psnrs)


def validation(val_loader, res, device, writer, step):
    res.eval()
    loss = []
    Psnr_list = []
    val_grid = torch.empty((16, 3, 64, 64))
    with torch.no_grad():
        for img, exp in tqdm.tqdm(val_loader):
            img = img.to(device)
            exp = exp.to(device).unsqueeze(0)
            for it in range(MAX_DEPTH):
                with torch.no_grad():
                    prob, z = res(img)
                action = torch.argmax(torch.tensor(prob))
                if action == STOP_ACTION:
                    break
                img = Actions.select(img, action).to('cuda')
            loss.append(torch.dist(img, exp, 2))
            Psnr_list.append(PSNR(img, exp))
            if len(loss) % 1 == 0:
                val_grid[int(len(loss) / 1) - 1] = img.squeeze()
    vpsnr = sum(Psnr_list) / len(Psnr_list)
    if writer is not None:
        writer.add_images('VAL IMAGE', val_grid, step)
        writer.add_scalar('L2 Validation Loss', sum(loss) / len(loss), step)
        writer.add_scalar('PSNR Validation Loss', vpsnr, step)
    print('L2 Validation Loss', (sum(loss) / len(loss)).item())
    res.train()
    return vpsnr


def main():
    args = parse_args()
    BASEDIR = args.basedir

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    warnings.filterwarnings("ignore")
    raw_dir = BASEDIR+'/TRAIN/low/'
    exp_dir = BASEDIR+'/TRAIN/high/'
    val_dirR = BASEDIR+'/VAL/low/'
    val_dirE = BASEDIR+'/VAL/high/'

    expname = args.expname
    weightfile = os.path.join("./", expname + ".pt")
    tblocation = os.path.join("./tensor/", expname)

    res = Network.ModifiedResnet(STOP_ACTION + 1, Dropout=args.dropout)
    res.to(device)

    images = args.num_images
    steps = args.num_steps
    val_images = args.val_images
    param = res.parameters()
    optimizer = torch.optim.AdamW(param, lr=args.lr)
    dataset = Dataset_LOL(raw_dir, exp_dir, size=args.size, training=True)

    val_set = Dataset_LOL(val_dirR, val_dirE, size=args.size, training=False)
    indices = random.sample(list(range(len(val_set))), val_images)
    val_set = torch.utils.data.Subset(val_set, indices)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1,
                                             drop_last=False,
                                             shuffle=True, num_workers=0)

    writer = SummaryWriter(tblocation)
    train_loss_history = collections.deque(maxlen=100)
    train_L1_history = collections.deque(maxlen=100)
    train_L2_history = collections.deque(maxlen=100)
    numGeneration = args.num_gen
    step = 0
    max_psnr = 0.0
    for gen_count in range(0, numGeneration + 1):
        samples = []
        indices = random.sample(list(range(len(dataset))), images)
        subset = torch.utils.data.Subset(dataset, indices)
        loader = torch.utils.data.DataLoader(subset, batch_size=args.bs, drop_last=False, shuffle=True, num_workers=0)
        print('GENERATION', gen_count)
        s, mean_error, psnr = generation(res, loader, steps, device)
        writer.add_scalar('L2 train Loss', mean_error, gen_count)
        writer.add_scalar('PSNR train Loss', psnr, gen_count)
        print('TRAIN')
        res, step = train(samples, res, optimizer, step, device,
                          writer, train_loss_history, train_L1_history,
                          train_L2_history,args, lambd=args.lambd)
        torch.save(res.state_dict(), weightfile)
        print('VALIDATION')
        if gen_count % 1 == 0:
            act_psnr = validation(val_loader, res, device, writer, gen_count)
            if act_psnr >= max_psnr:
                max_psnr = act_psnr
                best_model = res.state_dict()
                print('Best model updated', max_psnr)
                torch.save(best_model, './' + expname + '_best_model.pt')

