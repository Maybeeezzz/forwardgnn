"""Observe autograd saved storage lifetimes without an extra owning registry."""
import weakref


class PackedTensor:
    def __init__(self, tensor, owner):
        # Autograd owns this object instead of its usual saved tensor. Detach
        # avoids a graph reference cycle; data/storage are not copied.
        self.tensor = tensor.detach()
        self.owner = owner


class SavedMemory:
    def __init__(self, torch, owner):
        self.torch, self.owner = torch, owner
        self.live = weakref.WeakValueDictionary()
        self.sequence = 0
        self.context = None

    def pack(self, tensor):
        packed = PackedTensor(tensor, self.owner())
        self.live[self.sequence] = packed
        self.sequence += 1
        return packed

    @staticmethod
    def unpack(packed):
        return packed.tensor

    def start(self):
        self.context = self.torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack)
        self.context.__enter__()

    def roots(self):
        return [('autograd_saved.{}.{}'.format(p.owner, i), p.tensor)
                for i, p in list(self.live.items())]

    def close(self):
        if self.context is not None:
            self.context.__exit__(None, None, None)
            self.context = None
        self.live.clear()
