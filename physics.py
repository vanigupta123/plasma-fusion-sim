import numpy as np
import matplotlib.pyplot as plt

def dipole_moment(I, r, dir):
    return I * np.pi * r**2 * dir

def force_between_dipoles_eq(r, r_hat, m1, m2):
    mu_0 = 4e-7*np.pi
    return (3*mu_0/(4*np.pi*(r**4)))*((np.dot(m1, r_hat)*m2)+(np.dot(m2, r_hat)*m1)+(np.dot(m1, m2)*r_hat)-5*(np.dot(m1, r_hat)*np.dot(m2, r_hat)*r_hat))

def total_force(plasma_pos, coil_positions, coil_moments, m_plasma):
    # net force on the plasma from all coils
    # r_hat points coil -> plasma
    F = np.zeros(3)
    for coil_pos, m_coil in zip(coil_positions, coil_moments):
        sep = np.asarray(plasma_pos, dtype=float) - coil_pos
        r = np.linalg.norm(sep)
        F += force_between_dipoles_eq(r, sep / r, m_coil, m_plasma)
    return F

def jacobian(plasma_pos, coil_positions, coil_moments, m_plasma, eps=1e-4):
    # [[dFx/dx, dFx/dz],
    #  [dFz/dx, dFz/dz]]  via central differences
    # check dFx/dx < 0 (horizontal stability) and dFz/dz > 0 (vertical instability)
    plasma_pos = np.asarray(plasma_pos, dtype=float)
    J = np.zeros((2, 2))
    for col, axis in enumerate((0, 2)):  # x, z
        step = np.zeros(3)
        step[axis] = eps
        dF = (total_force(plasma_pos + step, coil_positions, coil_moments, m_plasma)
              - total_force(plasma_pos - step, coil_positions, coil_moments, m_plasma)) / (2 * eps)
        J[:, col] = dF[[0, 2]]
    return J

def integrator(plasma_pos, plasma_vel, coil_positions, coil_moments, m_plasma, mass, dt):
    # use euler integration to update plasma position and velocity
    F = total_force(plasma_pos, coil_positions, coil_moments, m_plasma)
    a = F / mass
    plasma_vel = plasma_vel + a * dt
    plasma_pos = plasma_pos + plasma_vel * dt
    return plasma_pos, plasma_vel

def coil_moments_from_currents(coil_currents, coil_radii):
    z_hat = np.array([0.0, 0.0, 1.0])  # every loop lies flat in the x-y plane, so moments point along z
    return np.array([dipole_moment(I, r, z_hat) for I, r in zip(coil_currents, coil_radii)])

def proportional_derivative_controller(z, z_vel, k_p, k_d, z_target=0.0):
    # delta_I_coil = -k_p * z_error - k_d * z_velocity
    # diverging coils should instead stabilize / have bounded oscillations
    z_error = z - z_target
    return -k_p * z_error - k_d * z_vel

def setup():
    # coil order: bottom, top, bottom-left, bottom-right, top-left, top-right
    c_bottom = (0, 0, -10)
    c_top = (0, 0, 10)
    c_diag_bl = (-10, 0, -10)
    c_diag_br = (10, 0, -10)
    c_diag_tl = (-10, 0, 10)
    c_diag_tr = (10, 0, 10)
    coil_positions = np.array([c_bottom, c_top, c_diag_bl, c_diag_br, c_diag_tl, c_diag_tr], dtype=float)
    coil_currents = np.array([10, 10, 10, 10, 10, 10], dtype=float)
    coil_radii = np.array([1, 1, 1, 1, 1, 1], dtype=float)
    m_plasma = dipole_moment(1, 1, np.array([0.0, 0.0, 1.0]))
    return coil_positions, coil_currents, coil_radii, m_plasma

def main():
    coil_positions, coil_currents, coil_radii, m_plasma = setup()
    coil_moments = coil_moments_from_currents(coil_currents, coil_radii)

    # instability check at equilibrium
    equilibrium = np.zeros(3)
    print("F at equilibrium:", total_force(equilibrium, coil_positions, coil_moments, m_plasma))
    J = jacobian(equilibrium, coil_positions, coil_moments, m_plasma)
    print("J =\n", J)
    print("horizontally stable:", J[0, 0] < 0, "| vertically unstable:", J[1, 1] > 0)

    # choose mass so the vertical growth rate gamma = sqrt(dFz/dz / mass) is 1 per unit time
    gamma_target = 1.0
    mass = J[1, 1] / gamma_target**2
    gamma = np.sqrt(J[1, 1] / mass)
    dt = 0.01 / gamma
    t_end = 6 / gamma

    # zero control, small initial vertical offset
    z0 = 0.01
    plasma = np.array([0.0, 0.0, z0])
    plasma_vel = np.zeros(3)

    times = []
    positions = []
    for i in range(int(t_end / dt)):
        times.append(i * dt)
        positions.append(plasma[2])  # record z position
        plasma, plasma_vel = integrator(plasma, plasma_vel, coil_positions, coil_moments, m_plasma, mass, dt)
    times = np.array(times)

    # starting at rest, the linearized solution is z0 * cosh(gamma * t)
    plt.semilogy(times, positions, label='simulated')
    plt.semilogy(times, z0 * np.cosh(gamma * times), '--', label=r'linear theory $z_0\cosh(\gamma t)$')
    plt.xlabel('time')
    plt.ylabel('plasma z position')
    plt.title('plasma position over time (zero control)')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
