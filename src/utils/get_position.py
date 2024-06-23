from src.dto.position import Position

def get_position(x1: int, x2: int, left_limit: int, center_limit: int):
    # Case 1: Both x-coordinates are to the left of the left limit
    if x1 < left_limit and x2 < left_limit:
        return Position.LEFT

    # Case 2: One x-coordinate is to the left and the other is in the center
    elif x1 < left_limit and x2 < center_limit:
        size_on_left_side = left_limit - x1
        size_on_center_side = x2 - left_limit

        # Determine which side is larger
        if size_on_left_side > size_on_center_side:
            return Position.CENTER_LEFT
        else:
            return Position.CENTER

    # Case 3: Both x-coordinates are within the center limits
    elif x1 < center_limit and x2 < center_limit:
        return Position.CENTER

    # Case 4: One x-coordinate is in the center and the other is on the right
    elif x1 < center_limit and x2 > center_limit:
        size_on_left_side = center_limit - x1
        size_on_center_side = x2 - center_limit

        # Determine which side is larger
        if size_on_left_side > size_on_center_side:
            return Position.CENTER_RIGHT
        else:
            return Position.CENTER

    # Case 5: Both x-coordinates are to the right of the center limit
    elif x1 > center_limit and x2 > center_limit:
        return Position.RIGHT

    # Case 6: Coordinates do not fit any of the above cases
    else:
        return Position.UNKNOWN
