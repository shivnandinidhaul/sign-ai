def landmarks_to_row(hand_landmarks):
    row = []

    for lm in hand_landmarks.landmark:
        row.extend([lm.x, lm.y, lm.z])

    return row
