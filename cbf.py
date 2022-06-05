#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import sympy
from numpy.typing import NDArray


@dataclass
class CBFBase:
    """CBF base class

    Attributes:
        G (NDArray): constraint matrix(=dh/dx)
        h (NDArray): constraint value(=h(x))
    Notes:
        The CBF optimization problem is formulated as follows
            minimize_{u} {cost function}
            subject to. G*u + alpha(h) <= 0
    """

    G: NDArray
    h: NDArray

    def get_constraints(self) -> Tuple[NDArray, NDArray]:
        """
        Returns:
            (NDArray): G
            (NDArray): alpha(h)
        """
        return self.G, self.alpha(self.h)

    def alpha(self, h: NDArray) -> NDArray:
        """
        Args:
            h (NDArray): constraint value(=h(x)). shape=(1,)
        Returns:
            (NDArray): h
        Notes:
            If you use specific alpha function, implement it with override.
        """
        return h


class GeneralCBF(CBFBase):
    def _calc_constranit_matrix(self, G: NDArray) -> None:
        self.G = G

    def _calc_constranit_value(self, h: NDArray) -> None:
        self.h = h

    def calc_constraints(self, G: NDArray, h: NDArray) -> None:
        self._calc_constranit_matrix(G)
        self._calc_constranit_value(h)


class Pnorm2dCBF(CBFBase):
    """
    Atrributes:
        cent_field (ndarray): center of field in world coordinate. shape=(2,)
        width (NDArray): half length of the major and minor axis of ellipse that match the x and y axis in the field coordinate. shape=(2,)
        theta (float): rotation angle(rad) world to the field coordinate.
        p (float): multiplier for p-norm.
        sign (float): 1->prohibit going outside of the field, -1->prohibit entering inside of the field.
        G (NDArray): constraint matrix(=dh/dx). shape=(2,)
        h (NDArray): constraint value(=h(x)). shape=(1,)
        x (sympy.Symbol): symbolic variable for cbf
        y (sympy.Symbol): symbolic variable for cbf
        cbf (sympy.Symbol): control barrier function
    Notes:
        If constraint value(=alpha(h(x))) satisfies h(x)>=0 then, the agent is inside of the ellipse
    """

    def __init__(self) -> None:
        self.x = sympy.Symbol("x", real=True)  # type: ignore
        self.y = sympy.Symbol("y", real=True)  # type: ignore

    def set_parameters(
        self,
        cent_field: NDArray,
        width: NDArray,
        theta: float = 0.0,
        p: float = 2.0,
        keep_inside: bool = True,
    ) -> None:
        """
        Args:
            cent_field (NDArray): center of field in world coordinate.
            width (NDArray): half length of the major and minor axis of ellipse that match the x and y axis in the field coordinate.
            theta (float): rotation angle(rad) world to the field coordinate. Defaults to 0.0.
            p (float): multiplier for p-norm. Defaults to 2.0.
            keep_inside (bool): flag to prohibit going outside of the field. Defaults to True.
        """
        self.cent_field = cent_field.flatten()
        self.width = width.flatten()
        self.theta = theta
        assert p >= 1
        self.p = p
        self.sign: int = 1 if keep_inside else -1

        self.cbf = 1.0 - sum(abs(np.array([self.x, self.y])) ** self.p) ** (1 / self.p)

    def get_parameters(self) -> Tuple[NDArray, NDArray, float, float, bool]:
        """
        Returns:
            Tuple[NDArray, NDArray, float, float, bool]: parameters
        """
        keep_inside = True if self.sign > 0 else False
        return self.cent_field, self.width, self.theta, self.p, keep_inside

    def _transform_agent_position(self, agent_position: NDArray) -> NDArray:
        """
        Args:
            agent_position (NDArray): agent position in world coordinate. shape=(2,)
        Returns:
            (NDArray): agent position in the normalized field coordinate. shape=(2,)
        Notes:
            The value for each axis is normalized by width.
        """
        rotation_matrix = self._calc_rotation_matrix(-self.theta)
        return rotation_matrix @ (agent_position - self.cent_field) / self.width

    def _calc_constraint_matrix(self, agent_position: NDArray) -> None:
        """
        Args:
            agent_position (NDArray): agent position in world coordinate. shape=(2,)
        Notes:
            coeff/self.width returns np.array([coeff[0]/self.width[0], coeff[1]/self.width[1]])
        """
        agent_position_transformed = self._transform_agent_position(agent_position)
        coeff = np.array(
            [
                self.cbf.diff(self.x).subs(  # type: ignore
                    [
                        (self.x, agent_position_transformed[0]),
                        (self.y, agent_position_transformed[1]),
                    ]
                ),
                self.cbf.diff(self.y).subs(  # type: ignore
                    [
                        (self.x, agent_position_transformed[0]),
                        (self.y, agent_position_transformed[1]),
                    ]
                ),
            ]
        )
        rotation_matrix = self._calc_rotation_matrix(self.theta)
        self.G = self.sign * rotation_matrix @ (coeff / self.width)

    def _calc_constraint_value(self, agent_position: NDArray) -> None:
        """
        Args:
            agent_position (NDArray): agent position in world coordinate. shape=(2,)
        """
        agent_position_transformed = self._transform_agent_position(agent_position)
        self.h = np.array(
            self.sign
            * self.cbf.subs(  # type: ignore
                [
                    (self.x, agent_position_transformed[0]),
                    (self.y, agent_position_transformed[1]),
                ]
            )
        )

    def calc_constraints(self, agent_position: NDArray) -> None:
        """
        Args:
            agent_position (NDArray): agent position in world coordinate.
        """
        agent_position = agent_position.flatten()
        self._calc_constraint_matrix(agent_position)
        self._calc_constraint_value(agent_position)

    @staticmethod
    def _calc_rotation_matrix(rad: float) -> NDArray:
        return np.array(
            [
                [np.cos(rad), -np.sin(rad)],
                [np.sin(rad), np.cos(rad)],
            ]
        )
