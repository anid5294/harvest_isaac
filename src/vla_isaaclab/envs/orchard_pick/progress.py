"""Feedback gates shared by the scripted strategy and simulator-free tests."""

PHASES = ("settle", "pregrasp", "approach", "close", "pull", "transfer",
          "lower", "release", "retreat", "hold")
MIN_STEPS = (45, 30, 30, 90, 45, 45, 45, 75, 45, 30)
MAX_STEPS = (90, 300, 300, 120, 300, 360, 300, 120, 300, 300)


class Progress:
    def __init__(self):
        self.index = 0
        self.steps = 0
        self.failure = None
        self.transitions = []

    @property
    def phase(self):
        return PHASES[self.index]

    def update(self, *, reached, detached, carried, grasp_contact=False):
        self.steps += 1
        gate = reached
        if self.phase in ("settle", "release"):
            gate = True
        elif self.phase == "close":
            gate = reached and grasp_contact
        elif self.phase == "pull":
            gate = reached and detached and carried
        elif self.phase == "hold":
            gate = False  # Only the environment's named success can finish the task.
        if gate and self.steps >= MIN_STEPS[self.index]:
            old = self.phase
            self.index += 1
            self.steps = 0
            self.transitions.append((old, self.phase))
        elif self.steps >= MAX_STEPS[self.index]:
            self.failure = "grasp_contact_missing" if self.phase == "close" else f"{self.phase}_timeout"
