"""Pure control-step gates for the attached-apple CPU trial."""

PHASES = (
    "settle", "pregrasp", "approach", "close", "verify_grasp", "pull",
    "lift", "transport", "lower", "release", "retreat", "verify_placement",
    "done", "failed",
)
TIMEOUT = {
    "settle": 60, "pregrasp": 360, "approach": 300, "close": 150,
    "verify_grasp": 90, "pull": 300, "lift": 300, "transport": 420,
    "lower": 360, "release": 150, "retreat": 300,
    "verify_placement": 300,
}


class HarvestFSM:
    """Advance only from measured task state, once per 30 Hz observation."""

    def __init__(self):
        self.phase = "settle"
        self.phase_steps = 0
        self.failure = None
        self.transitions = []
        self.contact_loss_steps = 0
        self.grasp_step = None
        self.break_step = None

    @property
    def failed(self):
        return self.failure is not None

    def _transition(self, phase, step):
        self.transitions.append({"from": self.phase, "to": phase, "step": step,
                                 "time_s": step / 30.0})
        self.phase = phase
        self.phase_steps = 0
        self.contact_loss_steps = 0

    def _fail(self, reason, step):
        self.failure = reason
        self._transition("failed", step)

    def update(self, *, step, metrics, ready):
        if self.phase in ("done", "failed"):
            return self.phase
        if metrics.get("failure"):
            self._fail(metrics["failure"], step)
            return self.phase
        detached = bool(metrics.get("detached"))
        contact = bool(metrics.get("grasp_contact"))
        if detached and self.break_step is None:
            self.break_step = step
            if self.grasp_step is None:
                self._fail("break_before_measured_grasp", step)
                return self.phase
            if self.phase != "pull":
                self._fail("break_before_pull", step)
                return self.phase
        self.phase_steps += 1
        if self.phase_steps > TIMEOUT[self.phase]:
            self._fail(f"{self.phase}_timeout", step)
            return self.phase
        if self.phase in ("pull", "lift", "transport", "lower"):
            self.contact_loss_steps = 0 if contact else self.contact_loss_steps + 1
            if self.contact_loss_steps > 15:
                self._fail("grasp_contact_lost", step)
                return self.phase
        phase = self.phase
        if phase == "settle" and self.phase_steps >= 20 and not detached:
            self._transition("pregrasp", step)
        elif phase == "pregrasp" and ready:
            self._transition("approach", step)
        elif phase == "approach" and ready and metrics.get("hand_distance", 1.0) < 0.16:
            self._transition("close", step)
        elif phase == "close" and contact and self.phase_steps >= 30:
            self.grasp_step = step
            self._transition("verify_grasp", step)
        elif phase == "verify_grasp" and contact and self.phase_steps >= 10:
            self._transition("pull", step)
        elif phase == "pull" and detached:
            # First observed native break ends pulling, regardless of pose.
            self._transition("lift", step)
        elif phase == "lift" and metrics.get("carried") and ready:
            self._transition("transport", step)
        elif phase == "transport" and metrics.get("carried") and ready:
            self._transition("lower", step)
        elif phase == "lower" and ready and metrics.get("in_release_xy"):
            self._transition("release", step)
        elif phase == "release" and self.phase_steps >= 35 and metrics.get("opened"):
            self._transition("retreat", step)
        elif phase == "retreat" and ready:
            self._transition("verify_placement", step)
        elif phase == "verify_placement" and metrics.get("success"):
            self._transition("done", step)
        return self.phase
