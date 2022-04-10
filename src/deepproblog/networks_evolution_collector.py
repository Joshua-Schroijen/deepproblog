from typing import Collection
from abc import ABC, abstractmethod

from .network import Network

class NetworksEvolutionCollector(ABC):
  @abstractmethod
  def collect_before_training(networks: Collection[Network]):
    pass

  @abstractmethod
  def collect_before_epoch(networks: Collection[Network]):
    pass

  @abstractmethod
  def collect_before_iteration(networks: Collection[Network]):
    pass

  @abstractmethod
  def collect_after_iteration(networks: Collection[Network]):
    pass
    
  @abstractmethod
  def collect_after_epoch(networks: Collection[Network]):
    pass

  @abstractmethod
  def collect_after_training(networks: Collection[Network]):
    pass