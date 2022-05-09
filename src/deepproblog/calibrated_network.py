# MIT License

# Copyright (c) 2017 Geoff Pleiss

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from abc import ABC, abstractmethod
from typing import Collection, Optional
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .network import Network
from .networks_evolution_collector import NetworksEvolutionCollector

class CalibratedNetwork(Network, ABC):
    class Options:
        @property
        def network_module(self):
            return self._network_module

        @property
        def name(self):
            return self._name

        @property
        def optimizer(self):
            return self._optimizer

        @property
        def scheduler(self):
            return self._scheduler

        @property
        def k(self):
            return self._k

        @property
        def batching(self):
            return self._batching

        @property
        def calibrate_after_each_train_iteration(self):
            return self._calibrate_after_each_train_iteration

        @network_module.setter
        def network_module(self, new):
            self._network_module = new

        @name.setter
        def name(self, new):
            self._name = new

        @optimizer.setter
        def optimizer(self, new):
            self._optimizer = new

        @scheduler.setter
        def scheduler(self, new):
            self._scheduler = new

        @k.setter
        def k(self, new):
            self._k = new

        @batching.setter
        def batching(self, new):
            self._batching = new

        @calibrate_after_each_train_iteration.setter
        def calibrate_after_each_train_iteration(self, new):
            self._calibrate_after_each_train_iteration = new

    @classmethod
    def fromOptions(
        cls,
        options: Options
    ):
        return cls(
            options.network_module,
            options.name,
            optimizer = options.optimizer,
            scheduler = options.scheduler,
            k = options.k,
            batching = options.batching,
            calibrate_after_each_train_iteration = options.calibrate_after_each_train_iteration
        )

    def __init__(
        self,
        network_module: torch.nn.Module,
        name: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler = None,
        k: Optional[int] = None,
        batching: bool = False,
        calibrate_after_each_train_iteration: bool = False
    ):
        super(Network, self).__init__(network_module, name, optimizer, scheduler, k, batching)
        self.uncalibrated_network_module = network_module
        self.calibrated_network_module = network_module
        self.calibrated = False
        self.calibrate_after_each_train_iteration = calibrate_after_each_train_iteration

        self._calibrate_called = False

    def eval(self):
        if self.calibrated == True:
            self.network_module = self.calibrated_network_module

        super().eval()

    def train(self):
        if self.calibrated == True:
            self.network_module = self.uncalibrated_network_module

        if self.calibrate_after_each_train_iteration == True:
            self.calibrate()

        super().train()

    @abstractmethod
    def calibrate(self):
        self._calibrate_called = True

        if self.eval_mode == True:
            self.network_module = self.calibrated_network_module

        self.calibrated = True

    def get_expected_calibration_error(self, valid_loader, n_bins = 15):
        with torch.no_grad():
            logits_list = []
            labels_list = []
            for input, label in valid_loader:
                if torch.cuda.is_available() == True:
                    input = input.cuda()

                logits = self.network_module(input)
                logits_list.append(logits)
                labels_list.append(label)

            if torch.cuda.is_available() == True:
                logits = torch.cat(logits_list).cuda()
                labels = torch.cat(labels_list).cuda()
            else:
                logits = torch.cat(logits_list)
                labels = torch.cat(labels_list)

            bin_boundaries = torch.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]

            softmaxes = F.softmax(logits, dim = 1)
            confidences, predictions = torch.max(softmaxes, 1)
            accuracies = predictions.eq(labels)

            ece = torch.zeros(1, device = logits.device)
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
                prop_in_bin = in_bin.float().mean()
                if prop_in_bin.item() > 0:
                    accuracy_in_bin = accuracies[in_bin].float().mean()
                    avg_confidence_in_bin = confidences[in_bin].mean()
                    ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            return ece.item()

    @property
    @abstractmethod
    def before_calibration_ece(self):
        pass

    @property
    @abstractmethod
    def after_calibration_ece(self):
        pass
      
class TemperatureScalingNetwork(CalibratedNetwork):
    class Options(CalibratedNetwork.Options):
        @property
        def valid_loader(self):
            return self._valid_loader

        @valid_loader.setter
        def valid_loader(self, new):
            self._valid_loader = new

    @classmethod
    def fromOptions(
        cls,
        options: Options
    ):
        return cls(
            options.network_module,
            options.name,
            options.valid_loader,
            optimizer = options.optimizer,
            scheduler = options.scheduler,
            k = options.k,
            batching = options.batching,
            calibrate_after_each_train_iteration = options.calibrate_after_each_train_iteration
        )

    def __init__(
        self,
        network_module: torch.nn.Module,
        name: str,
        valid_loader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler = None,
        k: Optional[int] = None,
        batching: bool = False,
        calibrate_after_each_train_iteration: bool = False
    ):
        super().__init__(network_module, name, optimizer, scheduler, k, batching, calibrate_after_each_train_iteration)
        self.valid_loader = valid_loader

    def calibrate(self):
        self.calibrated_network_module = _NetworkWithTemperature(self.uncalibrated_network_module).set_temperature(self.valid_loader)
        
        super().calibrate()

    @property
    def before_calibration_ece(self):
        if self._calibrate_called == True:
           return self.calibrated_network_module.before_temperature_ece
        else:
           return None

    @property
    def after_calibration_ece(self):
        if self._calibrate_called == True:
            return self.calibrated_network_module.after_temperature_ece
        else:
            return None

class _NetworkWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, valid_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        if torch.cuda.is_available() == True:
            self.cuda()
            nll_criterion = nn.CrossEntropyLoss().cuda()
            ece_criterion = _ECELoss().cuda()
        else:
            nll_criterion = nn.CrossEntropyLoss()
            ece_criterion = _ECELoss()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                if torch.cuda.is_available() == True:
                    input = input.cuda()

                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            if torch.cuda.is_available() == True:
                logits = torch.cat(logits_list).cuda()
                labels = torch.cat(labels_list).cuda()
            else:
                logits = torch.cat(logits_list)
                labels = torch.cat(labels_list)
        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        self.before_temperature_ece = before_temperature_ece
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        self.after_temperature_ece = after_temperature_ece
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        return self

    @property
    def before_temperature_ece(self):
        return self.before_temperature_ece

    @property
    def after_temperature_ece(self):
        return self.after_temperature_ece

class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins = 15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim = 1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device = logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

class NetworkECECollector(NetworksEvolutionCollector):
    def __init__(self, epoch_collect_iter: int = 1, iteration_collect_iter: int = 100):
        self.epoch_collect_iter = epoch_collect_iter
        self.iteration_collect_iter = iteration_collect_iter

        self.before_calibration_ece_history = {}
        self.after_calibration_ece_history = {}

        self._no_iterations = 0
        self._no_epochs = 0

    def collect_before_training(self, networks: Collection[Network]):
        pass

    def collect_before_epoch(self, networks: Collection[Network]):
        pass

    def collect_before_iteration(self, networks: Collection[Network]):
        pass

    def collect_after_iteration(self, networks: Collection[Network]):
        self._no_iterations += 1
        if self._no_iterations % self.iteration_collect_iter == 0:
            for name in networks:
                if isinstance(networks[name], CalibratedNetwork):
                    self.before_calibration_ece_history[name] = self.before_calibration_ece_history.get(name, [])
                    self.before_calibration_ece_history[name].append(networks[name].before_calibration_ece)
                    self.after_calibration_ece_history[name] = self.after_calibration_ece_history.get(name, [])
                    self.after_calibration_ece_history[name].append(networks[name].after_calibration_ece)

    def collect_after_epoch(self, networks: Collection[Network]):
        self._no_epochs += 1

    def collect_after_training(self, networks: Collection[Network]):
        pass
