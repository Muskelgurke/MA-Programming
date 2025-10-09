class EarlyStopping:
    def __init__(self, patience: int, delta: float):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None
        self.early_stop_inf = {}

    def check_validation(self, val_loss, model):
        if val_loss is None:
            self.early_stop = True
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)

    def set_break_info(self,info: dict) -> None:
        self.early_stop_info = info

    def break_cause_nan_is_loss(self,)->None:
        self.early_stop_nan_loss = True

    def get_break_info(self)->dict:
        return self.early_stop_info