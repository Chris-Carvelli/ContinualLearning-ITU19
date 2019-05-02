class Logger:
    def __init__(self, desc, total):
        self.desc = desc
        self.total = total

        self.count = 0

    def __enter__(self, *args, **kwargs):
        print(f'Starting {self.desc}')
        return self

    def __exit__(self, *args, **kwargs):
        print(f'Exiting {self.desc}')

    def step(self, log=None):
        print(f'\t[{self.count}/{self.total}]\t{log or ""}')
        self.count += 1
