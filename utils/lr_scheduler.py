class LR_Wramer():
    def __init__(self, optimizer, scheduler=None, until=2000):
        super().__init__()
        self.optimizer = optimizer
        self.base_lr = optimizer.param_groups[0]["lr"]
        self.until = until
        self.global_step = 0

        self.scheduler = scheduler

    def step(self, epoch=None):
        # warm up lr
        if self.global_step < self.until:
            lr_scale = min(1.0, float(
                self.global_step + 1) / self.until)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr_scale * self.base_lr
        else:
            if self.scheduler is not None:
                self.scheduler.step(epoch=epoch)

        self.global_step += 1
