import torch.nn
from typing import Dict


class ModelDict:
    def __init__(self, model_dict: Dict[str, Dict[str, torch.nn.Module]]):
        # Validate dictionary format
        if len(model_dict.keys()) == 0:
            raise TypeError("Incorrect ExperimentDict Structure")
        for ek in model_dict:
            if type(model_dict[ek]) is not dict:
                raise TypeError("Incorrect ExperimentDict Structure")
        # If the format is correct then store in the class
        self.__dict = model_dict

    def get_exp_list(self):
        return self.__dict.keys()

    def get_model_list(self, exp_name):
        return self.__dict[exp_name].keys()

    def set_train(self, exp_name, model_name):
        self.__dict[exp_name][model_name].train()

    def set_train_all(self):
        for e_exp_name in self.__dict.keys():
            for e_model_name in self.__dict[e_exp_name].keys():
                if self.__dict[e_exp_name][e_model_name] is not None:
                    self.__dict[e_exp_name][e_model_name].train()

    def set_eval(self, exp_name, model_name):
        self.__dict[exp_name][model_name].eval()

    def set_eval_all(self):
        for e_exp_name in self.__dict.keys():
            for e_model_name in self.__dict[e_exp_name].keys():
                if self.__dict[e_exp_name][e_model_name] is not None:
                    self.__dict[e_exp_name][e_model_name].eval()

    def get_model(self, exp_name, model_name):
        return self.__dict[exp_name][model_name]


class OptimDict:
    def __init__(self, optim_dict: Dict[str, Dict[str, torch.optim.Optimizer]]):
        # Validate dictionary format
        if len(optim_dict.keys()) == 0:
            raise TypeError("Incorrect ExperimentDict Structure")
        for ek in optim_dict:
            if type(optim_dict[ek]) is not dict:
                raise TypeError("Incorrect ExperimentDict Structure")
        # If the format is correct then store in the class
        self.__dict = optim_dict

    def get_exp_list(self):
        return self.__dict.keys()

    def get_optim_list(self, exp_name):
        return self.__dict[exp_name].keys()

    def zero_grad(self, exp_name, optim_name):
        if self.__dict[exp_name][optim_name] is not None:
            self.__dict[exp_name][optim_name].zero_grad()

    def zero_grad_all(self):
        for e_exp_name in self.__dict.keys():
            for e_optim_name in self.__dict[e_exp_name].keys():
                if self.__dict[e_exp_name][e_optim_name] is not None:
                    self.__dict[e_exp_name][e_optim_name].zero_grad()

    def step(self, exp_name, optim_name):
        if self.__dict[exp_name][optim_name] is not None:
            self.__dict[exp_name][optim_name].step()

    def step_all(self):
        for e_exp_name in self.__dict.keys():
            for e_optim_name in self.__dict[e_exp_name].keys():
                if self.__dict[e_exp_name][e_optim_name] is not None:
                    self.__dict[e_exp_name][e_optim_name].step()
