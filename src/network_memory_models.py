"""Small, explicit learning rules for the network/algorithm memory experiment.

Only PyTorch is required here; no GNN/PyG imports. CPU tests use the same code.
"""
from contextlib import nullcontext
import torch
from torch import nn
from torch.nn import functional as F


def mark(tensor, category, owner=None):
    tensor._memory_algorithm_category = category
    if owner is not None:
        tensor._memory_owner = owner
    return tensor


class Network(nn.Module):
    def __init__(self, architecture, algorithm, seed=100, lr=0.01):
        super().__init__()
        self.architecture, self.algorithm = architecture, algorithm
        # Extra heads/feedback never change the common backbone initialization.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            if architecture == 'mlp':
                self.layers = nn.ModuleList([nn.Sequential(nn.Linear(a, 256), nn.ReLU())
                                            for a in (784, 256, 256)])
                widths, flattened = [256]*3, [256]*3
            elif architecture == 'cnn':
                self.layers = nn.ModuleList([
                    nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.ReLU(), nn.AvgPool2d(2))
                    for a, b in ((1, 16), (16, 32), (32, 64))])
                widths, flattened = [16, 32, 64], [16*14*14, 32*7*7, 64*3*3]
            else:
                raise ValueError(architecture)
            self.head = nn.Linear(widths[-1], 10) if algorithm != 'ff' else None
            torch.manual_seed(seed+1000)
            self.local_heads = nn.ModuleList()
            if algorithm == 'local':
                self.local_heads.extend(nn.Linear(w, 10) for w in widths[:-1])
            if algorithm == 'dfa':
                for i, size in enumerate(flattened):
                    # Normalize by output class count; no spatial compression.
                    self.register_buffer('feedback_'+str(i), torch.randn(10, size)/10**0.5)
        self.lr = lr

    def setup(self, device):
        self.to(device)
        # .to() may replace buffer objects, so mark after device transfer.
        for i, head in enumerate(self.local_heads):
            for p in head.parameters():
                mark(p, 'local_heads', 'layer'+str(i+1))
        if self.algorithm == 'dfa':
            for i in range(3):
                mark(getattr(self, 'feedback_'+str(i)), 'feedback_state', 'layer'+str(i+1))
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        return self

    def input(self, x):
        return x.flatten(1) if self.architecture == 'mlp' else x

    @staticmethod
    def features(h):
        return h.mean((2, 3)) if h.ndim == 4 else h

    def forward(self, x, memory=None, step=0):
        h = self.input(x)
        for i, layer in enumerate(self.layers):
            with scope(memory, 'forward', i, step):
                h = layer(h)
        return self.head(self.features(h))


def scope(memory, phase, i=None, step=0):
    return (memory.stage(phase, layer='layers.'+str(i) if i is not None else None,
                         epoch=step, period='training') if memory else nullcontext())


def inject(x, labels):
    value = mark(x.clone(), 'algorithm_data')
    value.flatten(1)[:, :10] = F.one_hot(labels, 10).to(value.dtype)
    return value


def normalized(h):
    return h / h.flatten(1).norm(dim=1).clamp_min(1e-6).view(
        -1, *([1]*(h.ndim-1)))


def update(model, x, y, step=0, memory=None, generator=None, perturbation=0.001):
    """One macrostep: same minibatch, each algorithm's own update semantics."""
    algorithm = model.algorithm
    with scope(memory, 'zero_grad', step=step):
        model.optimizer.zero_grad(set_to_none=True)
    if algorithm == 'spsa':
        return spsa(model, x, y, step, memory, generator, perturbation)
    if algorithm == 'ff':
        return forward_forward(model, x, y, step, memory, generator)
    if algorithm == 'local':
        h, total = model.input(x), 0.0
        for i, layer in enumerate(model.layers):
            with scope(memory, 'forward', i, step):
                out = layer(h.detach())
            with scope(memory, 'objective', i, step):
                head = model.local_heads[i] if i < 2 else model.head
                logits = mark(head(model.features(out)), 'objective')
                loss = mark(F.cross_entropy(logits, y), 'objective')
            with scope(memory, 'local_backward', i, step):
                loss.backward()
            with scope(memory, 'optimizer_step', i, step):
                # Other blocks have grad=None, so this updates only this block/head.
                model.optimizer.step()
                model.optimizer.zero_grad(set_to_none=True)
            total += loss.item()
            h = mark(out.detach(), 'algorithm_cache', 'layer'+str(i+1))
            del out, logits, loss
        return total/3

    h, outputs = model.input(x), []
    for i, layer in enumerate(model.layers):
        with scope(memory, 'forward', i, step):
            h = layer(h.detach() if algorithm == 'dfa' else h)
            if algorithm == 'dfa':
                mark(h, 'algorithm_cache', 'layer'+str(i+1))
                outputs.append(h)
    with scope(memory, 'objective', 2, step):
        features = model.features(h.detach() if algorithm == 'dfa' else h)
        logits = mark(model.head(features), 'objective')
        loss = mark(F.cross_entropy(logits, y), 'objective')
    with scope(memory, 'backward', 2, step):
        loss.backward()
    if algorithm == 'dfa':
        with torch.no_grad(), scope(memory, 'feedback', step=step):
            error = mark((logits.softmax(-1)-F.one_hot(y, 10))/len(y), 'update_buffer')
        for i, out in enumerate(outputs):
            with torch.no_grad(), scope(memory, 'feedback', i, step):
                feedback = mark((error @ getattr(model, 'feedback_'+str(i))).reshape_as(out),
                                'update_buffer', 'layer'+str(i+1))
            with scope(memory, 'local_backward', i, step):
                out.backward(feedback)
            del feedback
    with scope(memory, 'optimizer_step', step=step):
        model.optimizer.step()
    return loss.item()


def forward_forward(model, x, y, step, memory, generator):
    with scope(memory, 'negative_data', step=step):
        negative = (y+torch.randint(1, 10, y.shape, device=y.device, generator=generator)) % 10
        pos = mark(model.input(inject(x, y)), 'algorithm_data')
        neg = mark(model.input(inject(x, negative)), 'algorithm_data')
    total = 0.0
    for i, layer in enumerate(model.layers):
        with scope(memory, 'forward_positive', i, step):
            out_pos = layer(pos.detach())
            good = mark(out_pos.flatten(1).square().mean(1), 'objective')
            loss = mark(0.5*F.softplus(2-good).mean(), 'objective')
        with scope(memory, 'local_backward_positive', i, step):
            loss.backward()
        total += loss.item()
        del loss, good
        with scope(memory, 'forward_negative', i, step):
            out_neg = layer(neg.detach())
            good = mark(out_neg.flatten(1).square().mean(1), 'objective')
            loss = mark(0.5*F.softplus(good-2).mean(), 'objective')
        with scope(memory, 'local_backward_negative', i, step):
            loss.backward()
        total += loss.item()
        with scope(memory, 'optimizer_step', i, step):
            model.optimizer.step()
            model.optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), scope(memory, 'representation', i, step):
            pos = mark(normalized(out_pos.detach()), 'algorithm_cache')
            neg = mark(normalized(out_neg.detach()), 'algorithm_cache')
        del out_pos, out_neg, loss, good
    return total/3


@torch.no_grad()
def spsa(model, x, y, step, memory, generator, c):
    parameters = list(model.parameters())
    with scope(memory, 'perturbation', step=step):
        backups = [mark(p.clone(), 'parameter_backup') for p in parameters]
        directions = [mark(torch.empty_like(p).bernoulli_(0.5, generator=generator).mul_(2).sub_(1),
                           'perturbation_state') for p in parameters]
    try:
        with scope(memory, 'perturbation_positive', step=step):
            for p, original, direction in zip(parameters, backups, directions):
                p.copy_(original).add_(direction, alpha=c)
            plus = F.cross_entropy(model(x, memory, step), y).item()
        with scope(memory, 'perturbation_negative', step=step):
            for p, original, direction in zip(parameters, backups, directions):
                p.copy_(original).add_(direction, alpha=-c)
            minus = F.cross_entropy(model(x, memory, step), y).item()
    finally:
        with scope(memory, 'restore_parameters', step=step):
            for p, original in zip(parameters, backups):
                p.copy_(original)
    with scope(memory, 'gradient_estimate', step=step):
        for p, direction in zip(parameters, directions):
            p.grad = mark(direction*((plus-minus)/(2*c)), 'gradients')
    with scope(memory, 'optimizer_step', step=step):
        model.optimizer.step()
    return (plus+minus)/2


@torch.no_grad()
def predict(model, x):
    if model.algorithm != 'ff':
        return model(x).argmax(1)
    scores = mark(torch.empty(len(x), 10, device=x.device), 'objective')
    for label in range(10):
        y = mark(torch.full((len(x),), label, device=x.device, dtype=torch.long), 'algorithm_data')
        h = mark(model.input(inject(x, y)), 'algorithm_data')
        total = mark(torch.zeros(len(x), device=x.device), 'objective')
        for layer in model.layers:
            h = layer(h)
            total += h.flatten(1).square().mean(1)
            h = mark(normalized(h), 'algorithm_cache')
        scores[:, label] = total
    return scores.argmax(1)
