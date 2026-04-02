from typing import Optional
from loguru import logger
import numpy as np
# Import the base Storage class from Vessim
from vessim.storage import Storage 

class BoundedSimpleBattery(Storage):
    """Custom battery with bounded state of charge (SoC) and optional C-rate limit."""

    def __init__(
        self,
        capacity: float,
        initial_soc: float = 0,
        min_soc: float = 0.2,  # Minimum SoC limit (e.g., 20%)
        max_soc: float = 0.8,  # Maximum SoC limit (e.g., 80%)
        c_rate: Optional[float] = None,
    ):
        self.capacity = capacity
        assert 0 <= initial_soc <= 1
        self.charge_level = capacity * initial_soc
        self._soc = initial_soc
        assert 0 <= min_soc <= 1
        assert 0 <= max_soc <= 1
        self.min_soc = min_soc
        self.max_soc = max_soc
        self.c_rate = c_rate
        self.total_discharged_Wh: float = 0.0  # Tracks lifetime discharged energy

    def update(self, power: float, duration: int) -> float:
        if duration <= 0.0:
            raise ValueError("Duration needs to be a positive value")
        if self._soc <= self.min_soc and power <= 0.0:
            return 0.0     
        if self._soc >= self.max_soc and power >= 0.0:
            return 0.0
        if self.c_rate is not None:
            max_power = self.c_rate * self.capacity
            if power >= max_power:
                power = max_power
            if power <= -max_power:
                power = -max_power

        charged_energy = power * duration
        new_charge_level = self.charge_level + charged_energy / 3600
        abs_min_soc = self.min_soc * self.capacity
        abs_max_soc = self.max_soc * self.capacity
        
        if new_charge_level < abs_min_soc:
            charged_energy = (abs_min_soc - self.charge_level) * 3600
            self.charge_level = abs_min_soc
            self._soc = self.min_soc
        elif new_charge_level > abs_max_soc:
            charged_energy = (abs_max_soc - self.charge_level) * 3600
            self.charge_level = abs_max_soc
            self._soc = self.max_soc
        else:
            self.charge_level = new_charge_level
            self._soc = self.charge_level / self.capacity
        if charged_energy < 0:
            # Convert the actual discharged Ws back to Wh and accumulate
            self.total_discharged_Wh += abs(charged_energy) / 3600

        return charged_energy

    def soc(self) -> float:
        return self._soc

    def state(self) -> dict:
        return {
            "soc": self._soc,
            "charge_level": self.charge_level,
            "capacity": self.capacity,
            "min_soc": self.min_soc,
            "max_soc": self.max_soc,
            "c_rate": self.c_rate,
            "cycles": self.total_discharged_Wh / self.capacity,
        }