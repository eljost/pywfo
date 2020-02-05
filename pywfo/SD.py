import numpy as np

class SD:

    def __init__(self, from_, to_, length, braket):
        self.from_ = from_
        self.to = to_
        self.length = length
        self.braket = braket

        mos = np.arange(length)
        mos[from_] = to_

        self.sign = (-1)**(length - self.from_ + 1)

    def __key(self):
        return (self.braket, self.from_, self.to)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return self.__key() == other.__key()

    def __str__(self):
        # return f"{self.sign: >2d}*SD({self.braket}, from={self.from_}, to={self.to})"
        return f"SD({self.from_}->{self.to}, {self.braket})"
    
    def __repr__(self):
        return self.__str__()
