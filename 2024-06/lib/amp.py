from torch.cuda.amp import GradScaler as GradScaler_
from torch.cuda.amp import autocast

class GradScaler(GradScaler_):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_scale_changed = False

    def update(self, **kwargs):
        scale_before = self.get_scale()
        super().update()
        scale_after = self.get_scale()
        self.is_scale_changed = scale_before > scale_after