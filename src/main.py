import calibrate


def main():
    calibrate.calibrate_camera()
    # goal_line_pts = select_goal_line_points()
    # plane = points_to_plane(goal_line_pts)
    # ball_pts = select_ball()
    # ball_distance = get_distance(ball_pts)
    # plane_distance = get_distance(plane)
    # if plane_distance < ball_distance:
    #     print("Goal!")
    # else:
    #     print("No Goal")


if __name__ == '__main__':
    main()
