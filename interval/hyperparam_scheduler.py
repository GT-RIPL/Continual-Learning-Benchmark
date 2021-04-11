class LinearScheduler:
    """
    class represent Linear Scheduler y = a * x
    """

    def __init__(self, start, end=None, coefficient=None):
        self.start = start
        self.end = end
        self.coefficient = coefficient
        self.current = start
        self.iteration = 0
        self.warm = None

    def step(self):
        assert self.coefficient is not None, "coefficient is None"

        if self.warm is not None:
            self.iteration += 1
            if self.iteration < self.warm:
                return self.current

        if self.end is None:
            self.current += self.coefficient
        else:
            if abs(self.current - self.end) > 1e-8:
                self.current += self.coefficient
            else:
                self.current = self.end

        return self.current

    def calc_coefficient(self, param_val, epoch, iter_on_epoch):
        self.coefficient = param_val / (epoch * iter_on_epoch)

    def warm_epoch(self, epoch, iter_on_epoch):
        self.warm = epoch * iter_on_epoch

    def set_end(self, end):
        self.end = None if end == 0 else end


if __name__ == '__main__':
    ls = LinearScheduler(start=1, end=0.5)
    ls.calc_coefficient(-0.5, 2, 200)
    for i in range(2 * 200):
        print(f"iter {i} - {ls.step()}")
    print(f"current: {ls.current}")
    print(ls.coefficient)
