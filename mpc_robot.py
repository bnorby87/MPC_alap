import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from mpcpy.utils import compute_path_from_wp
import mpcpy

P = mpcpy.Params()

import sys
import time

import rclpy
from motion import Motion


def get_state(rosbot):
    rclpy.spin_once(rosbot.robot)
    robPos = (rosbot.robot.pose.position.x, rosbot.robot.pose.position.y, rosbot.robot.pose.position.z)
    linVel = (rosbot.robot.velocity.linear.x, rosbot.robot.velocity.linear.y, rosbot.robot.velocity.linear.z)
    angVel = (rosbot.robot.velocity.angular.x, rosbot.robot.velocity.angular.y, rosbot.robot.velocity.angular.z)
    robOrn = rosbot.robot.theta
    
    return np.array(
        [
            robPos[0],
            robPos[1],
            np.sqrt(linVel[0] ** 2 + linVel[1] ** 2),
            robOrn
        ]
    )


def set_ctrl(rosbot, currVel, acceleration, steeringAngle):
    targetVelocity = currVel + acceleration * P.DT
    angularVelocity = steeringAngle * P.DT
    twist = rosbot.create_twist(lin_x = targetVelocity, ang_z = angularVelocity)
    rosbot._publisher.publish(twist)
    rclpy.spin_once(rosbot.robot)

def plot_results(path, x_history, y_history):
    """ """
    plt.style.use("ggplot")
    plt.figure()
    plt.title("MPC Tracking Results")

    plt.plot(
        path[0, :], path[1, :], c="tab:orange", marker=".", label="reference track"
    )
    plt.plot(
        x_history,
        y_history,
        c="tab:blue",
        marker=".",
        alpha=0.5,
        label="vehicle trajectory",
    )
    plt.axis("equal")
    plt.legend()
    plt.show()


def run_sim():

    # Interpolated Path to follow given waypoints
    path = compute_path_from_wp(
        [0, 3, 4, 6, 10, 11, 12, 6, 1, 0],
        [0, 0, 2, 4, 3, 3, -1, -6, -2, -2],
        P.path_tick,
    )

    rclpy.init(args=None)
    rosbot=Motion()
 
    # starting guess
    action = np.zeros(P.M)
    action[0] = P.MAX_ACC / 2  # a
    action[1] = 0.0  # delta

    # Cost Matrices
    Q = np.diag([20, 20, 10, 20])  # state error cost
    Qf = np.diag([30, 30, 30, 30])  # state final error cost
    R = np.diag([10, 10])  # input cost
    R_ = np.diag([10, 10])  # input rate of change cost

    mpc = mpcpy.MPC(P.N, P.M, Q, R)
    x_history = []
    y_history = []

    time.sleep(0.5)
    input("\033[92m Press Enter to continue... \033[0m")

    while 1:

        state = get_state(rosbot)
        x_history.append(state[0])
        y_history.append(state[1])

        if np.sqrt((state[0] - path[0, -1]) ** 2 + (state[1] - path[1, -1]) ** 2) < 0.2:
            print("Success! Goal Reached")
            set_ctrl(rosbot, 0, 0, 0, 0)
            plot_results(path, x_history, y_history)
            input("Press Enter to continue...")
            return

        # for MPC car ref frame is used
        state[0:2] = 0.0
        state[3] = 0.0

        # add 1 timestep delay to input
        state[0] = state[0] + state[2] * np.cos(state[3]) * P.DT
        state[1] = state[1] + state[2] * np.sin(state[3]) * P.DT
        state[2] = state[2] + action[0] * P.DT
        state[3] = state[3] + action[0] * np.tan(action[1]) / P.L * P.DT

        # optimization loop
        start = time.time()

        # State Matrices
        A, B, C = mpcpy.get_linear_model_matrices(state, action)

        # Get Reference_traj -> inputs are in worldframe
        target, _ = mpcpy.get_ref_trajectory(get_state(rosbot), path, 1.0)

        x_mpc, u_mpc = mpc.optimize_linearized_model(
            A, B, C, state, target, time_horizon=P.T, verbose=False
        )

        action[:] = [u_mpc.value[0, 1], u_mpc.value[1, 1]]

        elapsed = time.time() - start
        print("CVXPY Optimization Time: {:.4f}s".format(elapsed))

        set_ctrl(rosbot, state[2], action[0], action[1])

        if P.DT - elapsed > 0:
            time.sleep(P.DT - elapsed)


if __name__ == "__main__":
    run_sim()