from dataclasses import dataclass
import json


@dataclass
class ModelConfig:
    """Configuration parameters for Isolation Forest model"""

    n_estimators: int = 100
    contamination: float = 0.0000573
    random_state: int = 42

    @classmethod
    def from_json(cls, json_str: str) -> "ModelConfig":
        try:
            data = json.loads(json_str)
            return cls(**data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {str(e)}") from e

    def to_json(self):
        return json.dumps(self.__dict__)
