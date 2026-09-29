"""Explicit single-apple contact filters and conservative grasp evidence."""

FINGER_LINKS = (
    "left_hand_thumb_0_link", "left_hand_thumb_1_link", "left_hand_thumb_2_link",
    "left_hand_middle_0_link", "left_hand_middle_1_link",
    "left_hand_index_0_link", "left_hand_index_1_link",
)


def opposing_contact(forces, threshold=0.05):
    # Each magnitude is the filtered force between apple and named finger link.
    thumb = max(forces.get(name, 0.0) for name in FINGER_LINKS[:3])
    fingers = max(forces.get(name, 0.0) for name in FINGER_LINKS[3:])
    return thumb >= threshold and fingers >= threshold
