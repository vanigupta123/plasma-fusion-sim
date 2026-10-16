import numpy as np
import matplotlib.pyplot as plt

from physics import (setup, coil_moments_from_currents, total_force, jacobian,
                     integrator, proportional_derivative_controller)

# how the scalar PD output u is split across the coils (order: bottom, top, bl, br, tl, tr)
# coils attract the plasma, so u > 0 pulls it up: more current in the top coil, less in the bottom coil
CONTROL_PATTERN = np.array([-1.0, 1.0, 0.0, 0.0, 0.0, 0.0])

def control_gain(coil_positions, coil_currents, coil_radii, m_plasma, eps=1e-3):
    # B = dFz/du at equilibrium: how much vertical force one unit of u produces
    def Fz(u):
        moments = coil_moments_from_currents(coil_currents + u * CONTROL_PATTERN, coil_radii)
        return total_force(np.zeros(3), coil_positions, moments, m_plasma)[2]
    return (Fz(eps) - Fz(-eps)) / (2 * eps)

def simulate(z0, k_p, k_d, mass, dt, t_end, coil_positions, coil_currents, coil_radii, m_plasma):
    plasma = np.array([0.0, 0.0, z0])
    plasma_vel = np.zeros(3)
    times, positions, controls = [], [], []
    for i in range(int(t_end / dt)):
        # delta_I is a deviation from the nominal currents, recomputed each step (not accumulated)
        u = proportional_derivative_controller(plasma[2], plasma_vel[2], k_p, k_d)
        moments = coil_moments_from_currents(coil_currents + u * CONTROL_PATTERN, coil_radii)
        times.append(i * dt)
        positions.append(plasma[2])
        controls.append(u)
        plasma, plasma_vel = integrator(plasma, plasma_vel, coil_positions, moments, m_plasma, mass, dt)
    return np.array(times), np.array(positions), np.array(controls)

def main():
    coil_positions, coil_currents, coil_radii, m_plasma = setup()
    coil_moments = coil_moments_from_currents(coil_currents, coil_radii)

    # same normalization as milestone 0: mass chosen so the open-loop growth rate gamma = 1
    J = jacobian(np.zeros(3), coil_positions, coil_moments, m_plasma)
    mass = J[1, 1]
    gamma = np.sqrt(J[1, 1] / mass)
    dt = 0.01 / gamma
    t_end = 10 / gamma

    B = control_gain(coil_positions, coil_currents, coil_radii, m_plasma)
    print("dFz/dz =", J[1, 1], "| dFz/du =", B)

    # linearized closed loop: mass*z'' + B*k_d*z' + (B*k_p - dFz/dz)*z = 0
    # k_p must beat the instability (B*k_p > dFz/dz); k_d adds damping
    # gains are in units of dFz/dz / B, tune these two numbers by hand
    k_p = 3.0 * J[1, 1] / B
    k_d = 1.5 * J[1, 1] / B
    print("k_p =", k_p, "| k_d =", k_d)

    z0 = 0.01
    times, z_pd, u_pd = simulate(z0, k_p, k_d, mass, dt, t_end, coil_positions, coil_currents, coil_radii, m_plasma)
    _, z_open, _ = simulate(z0, 0.0, 0.0, mass, dt, t_end, coil_positions, coil_currents, coil_radii, m_plasma)
    print("final |z| with PD:", abs(z_pd[-1]), "| max |delta_I|:", np.max(np.abs(u_pd)))

    fig, (ax_z, ax_u) = plt.subplots(2, 1, sharex=True)
    ax_z.plot(times, z_pd, label='PD control')
    ax_z.plot(times, z_open, '--', label='zero control')
    ax_z.set_ylim(-2 * z0, 2 * z0)
    ax_z.set_ylabel('plasma z position')
    ax_z.set_title('plasma position over time (PD baseline)')
    ax_z.legend()
    ax_z.grid()
    ax_u.plot(times, u_pd)
    ax_u.set_xlabel('time')
    ax_u.set_ylabel(r'$\Delta I$ (top +, bottom -)')
    ax_u.grid()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
