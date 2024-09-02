from enum import Enum
from typing import Optional, Dict, Any


class State(Enum):
    INITIAL = 0
    EMERGENCY = 1
    MESSAGE = 2
    LOCATION = 3
    INTERMEDIARY = 4
    FINAL = 5


class StateManager:
    def __init__(self):
        self.current_state: State = State.INITIAL
        self.context: Dict[str, Any] = {}

    # update states and return a string for logging
    def transition_to(self, new_state: State) -> str:
        old_state = self.current_state
        self.current_state = new_state
        return f"State transitioned from {old_state.name} to {new_state.name}"

    # same for context
    def update_context(self, **kwargs) -> str:
        updates = []
        for key, value in kwargs.items():
            old_value = self.context.get(key, "None")
            self.context[key] = value
            updates.append(f"{key}: {old_value} -> {value}")
        return "Context updated: " + ", ".join(updates)

    def clear_context(self) -> str:
        self.context = {}
        return "Context cleared"

    def reset(self):
        self.__init__()

    @property
    def state(self) -> State:
        return self.current_state

    def get_context(self) -> dict:
        return {"state": self.current_state, **self.context}
