import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List
from dataclasses import dataclass


    
@dataclass(order=True)
class resonator: # not plural
    '''
    resonator class for holding resonator frequencies, fit parameters, and flags                                    
    '''
    frequency: float
    use: Optional[bool] = None
    flags: Optional[List[None]] = None

@dataclass()
class resonators_class: # plural
    resonators: List[resonator]

    def len_use_true(self):
        count = 0
        for resonator in self.resonators:
            if resonator.use == True:
                count += 1

        return count

    def all_frequencies(self):
        frequency_list =  []
        for resonator in self.resonators:
            frequency_list.append(resonator.frequency)

        return frequency_list
