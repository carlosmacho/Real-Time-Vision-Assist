from dataclasses import dataclass
from src.dto.position import Position


@dataclass
class Prediction:
    class_id: str
    position: Position

    def __str__(self):
        return f"{self.class_id} on {self.position.value}"

    def __hash__(self):
        return hash(f"{self.class_id}:{self.position}")