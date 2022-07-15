from typing import Collection
from abc import ABC, abstractmethod

from .network import Network

class NetworksEvolutionCollector(ABC):
  @abstractmethod
  def collection_as_dict(self):
    pass

  @abstractmethod
  def collect_before_training(self, networks: Collection[Network]):
    pass

  @abstractmethod
  def collect_before_epoch(self, networks: Collection[Network]):
    pass

  @abstractmethod
  def collect_before_iteration(self, networks: Collection[Network]):
    pass

  @abstractmethod
  def collect_after_iteration(self, networks: Collection[Network]):
    pass
    
  @abstractmethod
  def collect_after_epoch(self, networks: Collection[Network]):
    pass

  @abstractmethod
  def collect_after_training(self, networks: Collection[Network]):
    pass