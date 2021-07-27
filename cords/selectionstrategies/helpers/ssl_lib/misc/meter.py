class Meter:
    def __init__(self, ema_coef=0.9):
        self.ema_coef = ema_coef
        self.params = {}

    def add(self, params:dict, ignores:list = []):
        for k, v in params.items():
            if k in ignores:
                continue
            if not k in self.params.keys():
                self.params[k] = v
            else:
                self.params[k] -= (1 - self.ema_coef) * (self.params[k] - v)

    def state(self, header="", footer=""):
        state = header
        for k, v in self.params.items():
            state += f" {k} {v:.6g} |"
        return state + " " + footer

    def reset(self):
        self.params = {}
