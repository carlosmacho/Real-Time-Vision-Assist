from src.dto.position import Position


def get_position(x1: int, x2: int, left_limit: int, center_limit: int):
    if x1 < left_limit and x2 < left_limit:
        return Position.LEFT
    elif x1 < left_limit and x2 < center_limit:
        size_on_left_side = left_limit - x1
        size_on_center_side = x2 - left_limit

        if size_on_left_side > size_on_center_side:
            return Position.CENTER_LEFT
        else:
            return Position.CENTER
    elif x1 < center_limit and x2 < center_limit:
        return Position.CENTER
    elif x1 < center_limit:
        size_on_left_side = center_limit - x1
        size_on_center_side = x2 - center_limit

        if size_on_left_side > size_on_center_side:
            return Position.CENTER_RIGHT
        else:
            return Position.CENTER
    elif x1 > center_limit:
        return Position.RIGHT
    else:
        return Position.UNKNOWN