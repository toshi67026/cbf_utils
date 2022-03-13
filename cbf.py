#!/usr/bin/env python3

from typing import Tuple

import numpy as np
import sympy


class CBFBase:
    """CBF base class

    Attributes:
        G (np.ndarray): constraint matrix(=dh/dx)
        h (np.ndarray): constraint value(=h(x))
    Notes:
        The CBF optimization problem is formulated as follows
            minimize_{u} {cost function}
            subject to. G*u + alpha(h) <= 0
    """

    def __init__(self) -> None:
        self.G: np.ndarray = np.zeros(1)
        self.h: np.ndarray = np.zeros(1)

    def get_constraints(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            (np.ndarray): G
            (np.ndarray): alpha(h)
        """
        return self.G, self.alpha(self.h)

    def alpha(self, h: np.ndarray) -> np.ndarray:
        """
        Args:
            h (np.ndarray): constraint value(=h(x)). shape=(1,)
        Returns:
            (np.ndarray): h
        Notes:
            If you use specific alpha function, implement it with override.
        """
        return h


class Pnorm2dCBF(CBFBase):
    """
    Atrributes:
        cent_field (ndarray): center of field in world coordinate. shape=(2,)
        width (np.ndarray): half length of the major and minor axis of ellipse that match the x and y axis in the field coordinate. shape=(2,)
        theta (float): rotation angle(rad) world to the field coordinate.
        p (float): multiplier for p-norm.
        sign (float): 1->prohibit going outside of the field, -1->prohibit entering inside of the field.
        G (np.ndarray): constraint matrix(=dh/dx). shape=(1x2)
        h (np.ndarray): constraint value(=h(x)). shape=(1,)
        x (sympy.Symbol): symbolic variable for cbf
        y (sympy.Symbol): symbolic variable for cbf
        cbf (sympy.Symbol): control barrier function
    Notes:
        If constraint value(=alpha(h(x))) satisfies h(x)>=0 then, the agent is inside of the ellipse
    """

    def __init__(self) -> None:
        super().__init__()
        self.x = sympy.Symbol("x", real=True)  # type: ignore
        self.y = sympy.Symbol("y", real=True)  # type: ignore

    def set_parameters(
        self,
        cent_field: np.ndarray,
        width: np.ndarray,
        theta: float = 0.0,
        p: float = 2.0,
        keep_inside: bool = True,
    ) -> None:
        """
        Args:
            cent_field (np.ndarray): center of field in world coordinate.
            width (np.ndarray): half length of the major and minor axis of ellipse that match the x and y axis in the field coordinate.
            theta (float): rotation angle(rad) world to the field coordinate. Defaults to 0.0.
            p (float): multiplier for p-norm. Defaults to 2.0.
            keep_inside (bool): flag to prohibit going outside of the field. Defaults to True.
        """
        self.cent_field = cent_field.flatten()
        self.width = width.flatten()
        self.theta = theta
        self.p = p
        self.sign: int = 1 if keep_inside else -1

        # TODO(toshi) p = infの場合にmaxノルムになるような場合分けを実装
        self.cbf = 1.0 - (sum(abs(np.array([self.x, self.y])) ** self.p) ** (1 / self.p))

    def get_parameters(self) -> Tuple[np.ndarray, np.ndarray, float, float, bool]:
        """
        Returns:
            Tuple[np.ndarray, np.ndarray, float, float, bool]: parameters
        """
        keep_inside = True if self.sign > 0 else False
        return self.cent_field, self.width, self.theta, self.p, keep_inside

    def _transform_agent_position(self, agent_position: np.ndarray) -> np.ndarray:
        """
        Args:
            agent_position (np.ndarray): agent position in world coordinate. shape=(2,)
        Returns:
            (np.ndarray): agent position in the normalized field coordinate. shape=(2,)
        Notes:
            The value for each axis is normalized by width.
        """
        rotation_matrix = self._get_rotation_matrix(-self.theta)
        ret: np.ndarray = np.dot(rotation_matrix, agent_position - self.cent_field) / self.width
        return ret

    def _calc_constraint_matrix(self, agent_position: np.ndarray) -> None:
        """
        Args:
            agent_position (np.ndarray): agent position in world coordinate. shape=(2,)
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
        rotation_matrix = self._get_rotation_matrix(self.theta)
        self.G = self.sign * (np.dot(rotation_matrix, coeff / self.width))

    def _calc_constraint_value(self, agent_position: np.ndarray) -> None:
        """
        Args:
            agent_position (np.ndarray): agent position in world coordinate. shape=(2,)
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

    def calc_constraints(self, agent_position: np.ndarray) -> None:
        """
        Args:
            agent_position (np.ndarray): agent position in world coordinate.
        """
        agent_position = agent_position.flatten()
        self._calc_constraint_matrix(agent_position)
        self._calc_constraint_value(agent_position)

    @staticmethod
    def _get_rotation_matrix(rad: float) -> np.ndarray:
        return np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])


# TODO(toshi) その他のCBFクラス実装
