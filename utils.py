import os
import struct
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.transform import Rotation as R

merl_path = "./brdfs/"

BRDF_SAMPLING_RES_THETA_H = 90
BRDF_SAMPLING_RES_THETA_D = 90
BRDF_SAMPLING_RES_PHI_D = 360

RED_SCALE = 1.0 / 1500.0
GREEN_SCALE = 1.15 / 1500.0
BLUE_SCALE = 1.66 / 1500.0


def rotate_vector(vector, axis, angle):
    axis = axis / np.linalg.norm(axis)
    axis = np.tile(axis, (angle.shape[0], 1))
    angle = angle[:, np.newaxis]
    rotation = R.from_rotvec(axis * angle)
    rotated_vector = rotation.apply(vector)
    return rotated_vector


class MeasuredBRDF:
    def __init__(self, filename):
        self.filename = os.path.join(merl_path, filename)
        self.brdf = None
        self.load_brdf()

    def load_brdf(self):
        try:
            with open(self.filename, "rb") as file:
                # Read dimensions
                dims = struct.unpack("3i", file.read(12))  # Each int is 4 bytes
                n = dims[0] * dims[1] * dims[2]

                # Check dimensions
                if (
                    n
                    != BRDF_SAMPLING_RES_THETA_H
                    * BRDF_SAMPLING_RES_THETA_D
                    * BRDF_SAMPLING_RES_PHI_D
                    // 2
                ):
                    print("Dimensions don't match")
                    return False, None

                # Read BRDF data
                data = np.empty(3 * n, dtype=np.float32)
                for i in range(3 * n):
                    data[i] = struct.unpack("d", file.read(8))[0]
                data = data.reshape(
                    (
                        3,
                        BRDF_SAMPLING_RES_THETA_H,
                        BRDF_SAMPLING_RES_THETA_D,
                        BRDF_SAMPLING_RES_PHI_D // 2,
                    )
                )
                brdf_array = np.array(data)
                brdf_array[0, :, :, :] *= RED_SCALE
                brdf_array[1, :, :, :] *= GREEN_SCALE
                brdf_array[2, :, :, :] *= BLUE_SCALE
                brdf_array = brdf_array.transpose((1, 2, 3, 0))
                self.brdf = brdf_array
                theta_h_grid = (
                    np.linspace(0, 1, BRDF_SAMPLING_RES_THETA_H)
                    * BRDF_SAMPLING_RES_THETA_H
                )
                theta_d_grid = (
                    np.linspace(0, 1, BRDF_SAMPLING_RES_THETA_D)
                    * BRDF_SAMPLING_RES_THETA_D
                )
                phi_d_grid = (
                    np.linspace(0, 1, BRDF_SAMPLING_RES_PHI_D // 2)
                    * BRDF_SAMPLING_RES_PHI_D
                    // 2
                )
                self.interpolator = RegularGridInterpolator(
                    (theta_h_grid, theta_d_grid, phi_d_grid),
                    brdf_array,
                    method="linear",
                )

        except IOError:
            print("Could not open file")

    def half_diff_look_up_brdf(self, theta_h, theta_d, phi_d, use_interpolation=False):
        # Use half angle and difference angle to index the BRDF
        # input: theta_h, theta_d, phi_d in radians

        # theta_half is a non-linear mapping
        theta_half_deg = theta_h / (np.pi * 0.5) * BRDF_SAMPLING_RES_THETA_H
        id_theta_h = np.clip(
            np.sqrt(theta_half_deg * BRDF_SAMPLING_RES_THETA_H),
            0,
            BRDF_SAMPLING_RES_THETA_H - 1,
        )
        id_theta_d = np.clip(
            theta_d / (np.pi * 0.5) * BRDF_SAMPLING_RES_THETA_D,
            0,
            BRDF_SAMPLING_RES_THETA_D - 1,
        )
        id_phi_d = np.clip(
            phi_d / np.pi * BRDF_SAMPLING_RES_PHI_D / 2,
            0,
            BRDF_SAMPLING_RES_PHI_D // 2 - 1,
        )
        if use_interpolation:
            # return value interpolated between neighboring values
            if id_theta_d.shape == ():
                points = np.array([[id_theta_h, id_theta_d, id_phi_d]])
                return np.clip(self.interpolator(points)[0], 0, None)
            else:
                points = np.stack([id_theta_h, id_theta_d, id_phi_d], axis=1)
            return np.clip(self.interpolator(points), 0, None)
        else:
            # return the value with nearest index value, officially used in the Merl BRDF
            id_theta_h = (id_theta_h).astype(int)
            id_theta_d = (id_theta_d).astype(int)
            id_phi_d = (id_phi_d).astype(int)
            # print(id_theta_d, id_phi_d, id_theta_h)
            return np.clip(self.brdf[id_theta_h, id_theta_d, id_phi_d, :], 0, None)

    def std_coord_look_up_brdf(
        self, theta_i, theta_o, phi_i, phi_o, use_interpolation=False
    ):
        # Use standard coordinates to index the BRDF
        # input: theta_i, theta_o, phi_i, phi_o in radians

        # compute in vector

        in_vec_z = np.cos(theta_i)
        in_vec_x = np.sin(theta_i) * np.cos(phi_i)
        in_vec_y = np.sin(theta_i) * np.sin(phi_i)
        in_vec = np.vstack([in_vec_x, in_vec_y, in_vec_z]).transpose()
        in_vec = in_vec / np.linalg.norm(in_vec,axis=1, keepdims=True)

        # compute out vector

        out_vec_z = np.cos(theta_o)
        out_vec_x = np.sin(theta_o) * np.cos(phi_o)
        out_vec_y = np.sin(theta_o) * np.sin(phi_o)
        out_vec = np.vstack([out_vec_x, out_vec_y, out_vec_z]).transpose()
        out_vec = out_vec / np.linalg.norm(out_vec,axis=1, keepdims=True)

        # compute half vector
        half_x = (in_vec_x + out_vec_x) / 2
        half_y = (in_vec_y + out_vec_y) / 2
        half_z = (in_vec_z + out_vec_z) / 2
        half_vec = np.vstack([half_x, half_y, half_z]).transpose()
        half_vec = half_vec / np.linalg.norm(half_vec,axis=1, keepdims=True)

        # compute theta_h, theta_d, phi_d
        theta_h = np.arccos(half_vec[:,2])
        phi_h = np.arctan2(half_vec[:,1], half_vec[:,0])
        bi_normal = np.array([0, 1, 0])
        normal = np.array([0, 0, 1])

        tmp = rotate_vector(in_vec, normal, -phi_h)
        diff = rotate_vector(tmp, bi_normal, -theta_h)

        theta_d = np.arccos(diff[:,2])
        phi_d = np.arctan2(diff[:,1], diff[:,0])
        phi_d += np.where(phi_d < 0, np.pi, 0)
        
        theta_h = np.where(theta_h <= 0, 0, theta_h)
        
        
        return self.half_diff_look_up_brdf(theta_h, theta_d, phi_d, use_interpolation)


if __name__ == "__main__":
    mybrdf = MeasuredBRDF(os.path.join(merl_path, "cherry-235.binary"))
    
    # theta_h, theta_d, phi_d = 0.2, 0.1, 0.2
    # theta_h, theta_d, phi_d = theta_h * np.pi, theta_d * np.pi, phi_d * np.pi
    # theta_i, theta_o, phi_i, phi_o = 0.344, 0.14, 0.2, 0.4
    # theta_i, theta_o, phi_i, phi_o = theta_i * np.pi, theta_o * np.pi, phi_i * np.pi, phi_o * np.pi
    # print(mybrdf.half_diff_look_up_brdf(theta_h, theta_d, phi_d, use_interpolation=True))
    # print(mybrdf.half_diff_look_up_brdf(theta_h, theta_d, phi_d, use_interpolation=False))
    # print(mybrdf.std_coord_look_up_brdf(theta_i, theta_o, phi_i, phi_o, use_interpolation=True))
    # print(mybrdf.std_coord_look_up_brdf(theta_i, theta_o, phi_i, phi_o, use_interpolation=False))

    batch_size = 1000
    # theta_h = np.linspace(0, np.pi / 2, batch_size)
    # theta_d = np.linspace(0, np.pi / 2, batch_size)
    # phi_d = np.linspace(0, np.pi, batch_size)

    # print(
    #     mybrdf.half_diff_look_up_brdf(
    #         theta_h[109], theta_d[109], phi_d[109], use_interpolation=False
    #     )
    # )

    # # mybrdf.half_diff_look_up_brdf(theta_h,theta_d,phi_d,use_interpolation=True)
    # temp = mybrdf.half_diff_look_up_brdf(
    #     theta_h, theta_d, phi_d, use_interpolation=False
    # )

    # print(temp[109])
    # theta_i = np.linspace(0, np.pi / 2, batch_size)
    # phi_i = np.linspace(0, np.pi / 2, batch_size)
    # theta_o = np.linspace(0, np.pi * 2, batch_size)
    # phi_o = np.linspace(0, np.pi * 2, batch_size)
    
    # print(mybrdf.std_coord_look_up_brdf(theta_i[109], theta_o[109], phi_i[109], phi_o[109], use_interpolation=False))
    # temp = mybrdf.std_coord_look_up_brdf(
    #     theta_i, theta_o, phi_i, phi_o, use_interpolation=False
    # )
    # print(temp[109])
    # print("Done")
