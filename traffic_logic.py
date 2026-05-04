def update_signal(lane_counts):
    max_lane = lane_counts.index(max(lane_counts))

    vehicles = lane_counts[max_lane]

    # Dynamic timing
    if vehicles == 0:
        green_time = 5
    elif vehicles < 5:
        green_time = 10
    elif vehicles < 10:
        green_time = 20
    else:
        green_time = 30

    return max_lane, green_time