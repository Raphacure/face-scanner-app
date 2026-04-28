from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class User:
    data: Dict[str, Any]

    @classmethod
    def from_db_row(cls, row: Dict[str, Any]) -> "User":
        return cls(data=row)

    def to_dict(self) -> Dict[str, Any]:
        return self.data
