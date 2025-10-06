from enum import Enum

class CircuitBreaker(Enum):
    Unifei = 0
    Furnas = 1
    Tijuco = 2

class Actuation(Enum):
    Opening = 0
    Closing = 1