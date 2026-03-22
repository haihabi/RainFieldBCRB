import numpy as np
from enum import Enum


class ProjectionType(Enum):
    Trace = 0
    FieldMSE = 1
    FieldRMSE = 2


class Projection2MSE(object):
    def __init__(self, in_field, n, projection_type: ProjectionType = ProjectionType.Trace):
        self.in_field = in_field
        self.n = n
        self.projection_type = projection_type
        if self.projection_type in [ProjectionType.FieldMSE, ProjectionType.FieldRMSE]:
            self.projection_matrix = in_field.projection_matrix(self.n)

    def actual_points(self):
        if self.projection_type in [ProjectionType.FieldMSE, ProjectionType.FieldRMSE]:
            return self.projection_matrix.shape[0]
        else:
            return int(np.sqrt(len(self.in_field.x_set) * len(self.in_field.y_set)))

    def __call__(self, in_crb):
        if self.projection_type == ProjectionType.Trace:
            return np.trace(in_crb) / in_crb.shape[0], in_crb
        elif self.projection_type == ProjectionType.FieldMSE:
            crb_proj = self.projection_matrix.T @ in_crb @ self.projection_matrix
            return np.trace(crb_proj) / crb_proj.shape[0], crb_proj
        elif self.projection_type == ProjectionType.FieldRMSE:
            crb_proj = self.projection_matrix.T @ in_crb @ self.projection_matrix
            return np.sqrt(np.trace(crb_proj) / crb_proj.shape[0]), crb_proj
        else:
            raise NotImplemented


class ProjectionMCRB2MSE(object):
    def __init__(self, project2mse):
        self.project2mse = project2mse

    def __call__(self, in_mcrb, in_theta_zero, in_theta):

        if self.project2mse.projection_type == ProjectionType.Trace:
            r = in_theta - in_theta_zero
            mcrb = in_mcrb
        elif self.project2mse.projection_type == ProjectionType.FieldMSE:
            h_proj = self.project2mse.projection_matrix.T
            r = h_proj @ (in_theta - in_theta_zero)
            mcrb = h_proj @ in_mcrb @ h_proj.T

        rrt = np.expand_dims(r, axis=1) @ np.expand_dims(r, axis=0)

        return np.trace(mcrb + rrt) / mcrb.shape[0]
        pass
