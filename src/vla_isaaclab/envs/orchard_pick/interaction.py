"""Measured commercial-harvest evidence, independent of controller phases."""

from dataclasses import dataclass


@dataclass
class HarvestEvidence:
    contact_age: int = 1000000
    detached_seen: bool = False
    lift_steps: int = 0
    carried: bool = False
    failure: str | None = None

    def update(self, *, detached, sustained_contact, lift_candidate):
        self.contact_age = 0 if sustained_contact else self.contact_age+1
        recent_contact = self.contact_age <= 2
        if detached and not self.detached_seen:
            if not recent_contact:
                self.failure = "detached_without_sustained_grasp"
            self.detached_seen = True
        self.lift_steps = (self.lift_steps+1 if detached and lift_candidate
                           and recent_contact else 0)
        self.carried |= self.lift_steps >= 10 and self.failure is None

    def released(self, *, opened, contact, hand_distance):
        return bool(self.carried and opened and not contact and hand_distance > 0.18)
