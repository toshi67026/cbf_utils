#!/usr/bin/env python3

import math
import os
import sys

import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from typing import List, Tuple

import numpy as np
from cbf import Pnorm2dCBF
from cbf_qp_solver import CBFNomQPSolver
from matplotlib import patches
from matplotlib.animation import FuncAnimation


class FieldCBFOptimizer:
    def __init__(self) -> None:
        self.qp_nom_solver = CBFNomQPSolver()
        self.P = np.eye(2)
        self.G_list: List[np.ndarray] = []
        self.alpha_h_list: List[np.ndarray] = []

        self.field_cbf = Pnorm2dCBF()

        # Initialize field (must be overwritten)
        cent_field = np.zeros(2)
        width: np.ndarray = np.array([3.0, 2.0])
        self.set_field_parameters(cent_field, width)

    def set_field_parameters(
        self, cent_field: np.ndarray, width: np.ndarray, theta: float = 0.0, p: float = 2.0, keep_inside: bool = True
    ) -> None:
        self.field_cbf.set_parameters(cent_field, width, theta, p, keep_inside)

    def get_field_parameters(self) -> Tuple[np.ndarray, np.ndarray, float, float, bool]:
        return self.field_cbf.get_parameters()

    def calc_field_constraints(self, agent_position: np.ndarray) -> None:
        cent_field, width, theta, p, keep_inside = self.get_field_parameters()
        self.field_cbf.set_parameters(cent_field, width, theta, p, keep_inside)
        self.field_cbf.calc_constraints(agent_position)

    def get_field_constraints(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.field_cbf.get_constraints()

    def append_constraints(
        self, G_list: List[np.ndarray], alpha_h_list: List[np.ndarray], G: np.ndarray, alpha_h: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        G_list.append(G)
        alpha_h_list.append(alpha_h)
        return G_list, alpha_h_list

    def set_qp_problem(self) -> None:
        G_list: List[np.ndarray] = []
        alpha_h_list: List[np.ndarray] = []

        G, alpha_h = self.get_field_constraints()
        G_list, alpha_h_list = self.append_constraints(G_list, alpha_h_list, G, alpha_h)

        self.alpha_h_list = alpha_h_list
        self.G_list = G_list

    def optimize(self, nominal_input: np.ndarray, agent_position: np.ndarray) -> Tuple[str, np.ndarray]:
        self.calc_field_constraints(agent_position)
        self.set_qp_problem()

        try:
            return self.qp_nom_solver.optimize(nominal_input, self.P, self.G_list, self.alpha_h_list)
        except Exception as e:
            raise e


def main() -> None:
    optimizer = FieldCBFOptimizer()

    initial_agent_position: np.ndarray = np.array([2, 0])
    agent_position_list: List[np.ndarray] = [initial_agent_position]
    dt = 0.1

    fig, ax = plt.subplots()

    def update(
        frame: int,
        agent_position_list: List[np.ndarray],
    ) -> None:
        ax.cla()

        cent_field: np.ndarray = np.array([np.cos(frame // 200), np.sin(frame // 200)])
        width: np.ndarray = np.array([1.5, 1])

        theta = 0.0
        p = 2.0

        # switch keep_inside flag
        if (frame // 100) % 2 == 0:
            keep_inside = True
        else:
            keep_inside = False

        optimizer.set_field_parameters(cent_field, width, theta, p, keep_inside)
        nominal_input: np.ndarray = np.array([np.cos(frame / 10), np.sin(frame / 10)])
        agent_position = agent_position_list[-1]
        _, optimal_input = optimizer.optimize(nominal_input, agent_position)

        # show vector
        ax.quiver(agent_position[0], agent_position[1], nominal_input[0], nominal_input[1], color="black")
        ax.quiver(agent_position[0], agent_position[1], optimal_input[0], optimal_input[1], color="red")

        ax.plot(agent_position[0], agent_position[1], "o", linewidth=5, label="agent")
        ax.plot([0], [0], linewidth=5, color="black", label="nominal_input")
        ax.plot([0], [0], linewidth=5, color="red", label="optimal_input")
        ax.plot([0], [0], linewidth=5, color="green", alpha=0.3, label="keep_inside: " + str(keep_inside))
        agent_position_list.append(agent_position + optimal_input.flatten() * dt)

        # show field
        r = patches.Rectangle(
            xy=[
                cent_field[0] - width[0] * np.cos(theta) + width[1] * np.sin(theta),
                cent_field[1] - width[0] * np.sin(theta) - width[1] * np.cos(theta),
            ],
            width=width[0] * 2,
            height=width[1] * 2,
            angle=theta * 180 / math.pi,
            color="green",
            alpha=0.3,
        )
        ax.add_patch(r)

        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid()
        ax.legend(loc="upper right")

        lim = [-5, 5]
        plt.xlim(lim)
        plt.ylim(lim)

    ani = FuncAnimation(
        fig,
        update,
        frames=1000,
        fargs=(agent_position_list,),
        interval=10,
        repeat=False,
    )

    # ani.save("field_cbf.gif")
    plt.show()


if __name__ == "__main__":
    main()
