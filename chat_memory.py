from typing import List, Tuple


class ChatMemory:
    def __init__(self, max_turns: int = 4):
        """
        max_turns: how many previous exchanges to keep (each exchange = one user + one bot).
        Example: max_turns=3 keeps last 3 exchanges.
        """
        if max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        self.max_turns = max_turns
        self._buffer: List[Tuple[str, str]] = []  

    def add_turn(self, user_text: str, bot_text: str):
        """Add a completed exchange to memory and drop oldest if overflowing."""
        self._buffer.append((user_text.strip(), bot_text.strip()))
    
        if len(self._buffer) > self.max_turns:
            self._buffer = self._buffer[-self.max_turns :]

    def get_context_prompt(self, current_user_input: str) -> str:
        """
        Build a prompt combining recent exchanges + current user input.
        Format:
          User: ...
          Bot: ...
          ...
          User: <current_user_input>
          Bot:
        The generator should continue from 'Bot:'.
        """
        lines = []
        for u, b in self._buffer:
            lines.append(f"User: {u}")
            lines.append(f"Bot: {b}")
        lines.append(f"User: {current_user_input.strip()}")
        lines.append("Bot:")
        return "\n".join(lines)

    def clear(self):
        self._buffer.clear()

    def __len__(self):
        return len(self._buffer)

    def __repr__(self):
        return f"<ChatMemory turns={len(self._buffer)}/{self.max_turns}>"
