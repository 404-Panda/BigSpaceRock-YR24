import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import datetime
import random
from scipy.optimize import fsolve
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import csv
import pygame
import sys

plt.style.use('dark_background')

# Constants
G = 6.67430e-11  # m^3 kg^-1 s^-2
M_SUN = 1.989e30  # kg
M_EARTH = 5.972e24  # kg
M_MOON = 7.342e22  # kg
AU_TO_M = 1.496e11  # meters per AU
DAY_TO_S = 86400  # seconds per day
R_EARTH_M = 6371000  # Earth radius in meters

# Simulation Setup
start_date = datetime.date(2025, 1, 1)
impact_date = datetime.date(2037, 12, 22)
delta_days = (impact_date - start_date).days
FRAMES = 600
INTERVAL = 40
t_days = np.linspace(0, delta_days, FRAMES)
dt = delta_days / FRAMES  # Time step in days

# Orbital Parameters
PERIOD_EARTH = 365.25
PERIOD_MOON = 27.3
PERIOD_ASTEROID = 4 * PERIOD_EARTH  # 4-year orbit
omega_earth = 2 * np.pi / PERIOD_EARTH
omega_moon = 2 * np.pi / PERIOD_MOON
omega_asteroid = 2 * np.pi / PERIOD_ASTEROID
R_EARTH_ORBIT = 1.0  # AU
R_MOON_ORBIT = 0.00257  # AU
a = 1.5  # Asteroid semi-major axis (AU)
b = 0.5  # Semi-minor axis (AU)
AST_INCLINATION = np.radians(random.uniform(0, 10))
AST_ASC_NODE = np.radians(random.uniform(0, 360))

# Initial Velocities (AU/day)
mu = G * M_SUN * DAY_TO_S**2 / AU_TO_M**3  # Gravitational parameter in AU/day units
v_asteroid = np.sqrt(mu * (2 / a - 1 / a))  # Vis-viva equation at perihelion
EARTH_V0 = np.array([0.0, 2 * np.pi * R_EARTH_ORBIT / PERIOD_EARTH, 0.0])
MOON_V0 = np.array([0.0, 2 * np.pi * R_MOON_ORBIT / PERIOD_MOON, 0.0]) + EARTH_V0
AST_V0 = np.array([0.0, v_asteroid, 0.0])  # Start at perihelion

# Asteroid Properties
AST_DIAMETER_M = 40.0
AST_DENSITY = 3000.0  # kg/m^3
AST_RADIUS_M = AST_DIAMETER_M / 2
AST_VOLUME = (4/3) * np.pi * (AST_RADIUS_M**3)
AST_MASS = AST_DENSITY * AST_VOLUME

# Physics Functions
def gravitational_acceleration(pos, mass, other_pos):
    r_vec = pos - other_pos
    r = np.linalg.norm(r_vec) * AU_TO_M
    if r < 1e-10:
        return np.zeros(3)
    return -G * mass / (r**3) * r_vec * AU_TO_M  # m/s^2

def atmospheric_drag(pos_earth, pos_ast, vel_ast):
    dist = np.linalg.norm(pos_earth - pos_ast) * AU_TO_M
    if dist < R_EARTH_M * 2:
        rho = 1.225 * np.exp(-dist / (R_EARTH_M * 0.1))
        area = np.pi * (AST_RADIUS_M**2)
        Cd = 2.0
        v = np.linalg.norm(vel_ast) * AU_TO_M / DAY_TO_S
        drag = -0.5 * rho * v**2 * Cd * area / AST_MASS
        return drag * vel_ast / v * DAY_TO_S**2 / AU_TO_M
    return np.zeros(3)

def update_positions_velocities(pos_e, pos_m, pos_a, vel_e, vel_m, vel_a, dt):
    acc_e = (gravitational_acceleration(pos_e, M_SUN, np.zeros(3)) +
             gravitational_acceleration(pos_e, M_MOON, pos_m))
    acc_m = (gravitational_acceleration(pos_m, M_SUN, np.zeros(3)) +
             gravitational_acceleration(pos_m, M_EARTH, pos_e))
    acc_a = (gravitational_acceleration(pos_a, M_SUN, np.zeros(3)) +
             gravitational_acceleration(pos_a, M_EARTH, pos_e) +
             gravitational_acceleration(pos_a, M_MOON, pos_m) +
             atmospheric_drag(pos_e, pos_a, vel_a))

    vel_e += acc_e * dt * DAY_TO_S**2 / AU_TO_M
    vel_m += acc_m * dt * DAY_TO_S**2 / AU_TO_M
    vel_a += acc_a * dt * DAY_TO_S**2 / AU_TO_M

    pos_e += vel_e * dt
    pos_m += vel_m * dt
    pos_a += vel_a * dt

    return pos_e, pos_m, pos_a, vel_e, vel_m, vel_a

# Initial Positions
def initial_earth_pos(t): return np.array([R_EARTH_ORBIT * np.cos(omega_earth * t), R_EARTH_ORBIT * np.sin(omega_earth * t), 0.0])
def initial_moon_pos(t):
    pos_e = initial_earth_pos(t)
    return pos_e + np.array([R_MOON_ORBIT * np.cos(omega_moon * t), R_MOON_ORBIT * np.sin(omega_moon * t), 0.0])
def initial_asteroid_pos(t, phase):
    x = a * np.cos(omega_asteroid * t + phase)
    y = b * np.sin(omega_asteroid * t + phase)
    z = 0.0
    x_temp = x
    y_temp = y * np.cos(AST_INCLINATION) - z * np.sin(AST_INCLINATION)
    z_temp = y * np.sin(AST_INCLINATION) + z * np.cos(AST_INCLINATION)
    return np.array([x_temp * np.cos(AST_ASC_NODE) - y_temp * np.sin(AST_ASC_NODE),
                     x_temp * np.sin(AST_ASC_NODE) + y_temp * np.cos(AST_ASC_NODE), z_temp])

# Collision Phase Adjustment
T = delta_days
def collision_equation(phase):
    pos_e = initial_earth_pos(T)
    pos_a = initial_asteroid_pos(T, phase)
    return np.linalg.norm(pos_e - pos_a)**2
phase_solution = fsolve(collision_equation, 0.0)[0]

# Figure Setup
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('black')
ax.grid(False)
ax.set_xlim(-2, 2)  # Expanded to see asteroid orbit
ax.set_ylim(-2, 2)
ax.set_zlim(-0.5, 0.5)
ax.axis('off')

# Starry Background
num_stars = 500
max_radius = 1.5 * a
star_r = max_radius * np.cbrt(np.random.rand(num_stars))
star_theta = 2 * np.pi * np.random.rand(num_stars)
star_phi = np.pi * np.random.rand(num_stars)
star_x = star_r * np.sin(star_phi) * np.cos(star_theta)
star_y = star_r * np.sin(star_phi) * np.sin(star_theta)
star_z = star_r * np.cos(star_phi)
ax.scatter(star_x, star_y, star_z, s=1, c='white', alpha=0.7)
sun = ax.scatter(0, 0, 0, s=200, c='yellow', marker='o', label='Sun')

# Ecliptic Plane
ecliptic_x, ecliptic_y = np.meshgrid(np.linspace(-2, 2, 20), np.linspace(-2, 2, 20))
ecliptic_z = np.zeros_like(ecliptic_x)
ax.plot_surface(ecliptic_x, ecliptic_y, ecliptic_z, color='gray', alpha=0.2)

# Reference Orbits
orbit_points = 300
theta_vals = np.linspace(0, 2*np.pi, orbit_points)
x_orbit_earth = R_EARTH_ORBIT * np.cos(theta_vals)
y_orbit_earth = R_EARTH_ORBIT * np.sin(theta_vals)
z_orbit_earth = np.zeros_like(theta_vals)
ax.plot(x_orbit_earth, y_orbit_earth, z_orbit_earth, c='blue', ls='--', alpha=0.5)
moon_orbit_line, = ax.plot([], [], [], c='gray', ls='--', alpha=0.3)
x_orbit_ast = a * np.cos(theta_vals)
y_orbit_ast = b * np.sin(theta_vals)
z_orbit_ast = np.zeros_like(theta_vals)
x_ref = x_orbit_ast * np.cos(AST_ASC_NODE) - y_orbit_ast * np.sin(AST_ASC_NODE)
y_ref = x_orbit_ast * np.sin(AST_ASC_NODE) + y_orbit_ast * np.cos(AST_ASC_NODE)
y_ref_tilt = y_ref * np.cos(AST_INCLINATION)
z_ref_tilt = y_ref * np.sin(AST_INCLINATION)
ax.plot(x_ref, y_ref_tilt, z_ref_tilt, c='green', ls='--', alpha=0.5)

# Markers and Trails
earth_marker = ax.scatter([], [], [], color='blue', s=50, label='Earth')
moon_marker = ax.scatter([], [], [], color='white', s=20, label='Moon')
asteroid_marker = ax.scatter([], [], [], color='green', s=60, label='Asteroid')
earth_trail = []
moon_trail = []
asteroid_trail = []
max_trail_length = 100  # Increased for visibility
earth_trail_line, = ax.plot([], [], [], 'b-', lw=1, alpha=0.5)
moon_trail_line, = ax.plot([], [], [], 'w-', lw=1, alpha=0.5)
asteroid_trail_line, = ax.plot([], [], [], 'g-', lw=1, alpha=0.5)

# Earth Surface
earth_lat = np.linspace(-np.pi/2, np.pi/2, 50)
earth_lon = np.linspace(0, 2*np.pi, 50)
earth_x_rot = np.cos(earth_lat) * np.cos(earth_lon[:, None])
earth_y_rot = np.cos(earth_lat) * np.sin(earth_lon[:, None])
earth_z_rot = np.sin(earth_lat) * np.ones_like(earth_lon[:, None])
earth_surface, = ax.plot([], [], [], c='cyan', alpha=0.2)

# Cardinal Directions
cardinal_scale = 0.1
north = ax.quiver(0, 0, 0, 0, 0, cardinal_scale, color='red', alpha=0.5)
south = ax.quiver(0, 0, 0, 0, 0, -cardinal_scale, color='red', alpha=0.5)
east = ax.quiver(0, 0, 0, cardinal_scale, 0, 0, color='white', alpha=0.5)
west = ax.quiver(0, 0, 0, -cardinal_scale, 0, 0, color='white', alpha=0.5)

# Text Panels
impact_info_text = ax.text2D(0.05, 0.92, "Asteroid 2024 YR4\nCollision: Dec 22, 2037",
                             transform=ax.transAxes, color='red', fontsize=9)
countdown_text = ax.text2D(0.5, 0.05, "", transform=ax.transAxes, color='yellow', fontsize=10, ha='center')
info_panel = ax.text2D(0.02, 0.02, "", transform=ax.transAxes, color='cyan', fontsize=8,
                       bbox=dict(facecolor='black', alpha=0.5, edgecolor='cyan'))
physics_panel = ax.text2D(0.75, 0.75, "", transform=ax.transAxes, color='magenta', fontsize=8, ha='right',
                          bbox=dict(facecolor='black', alpha=0.5, edgecolor='magenta'))
impact_text = ax.text2D(0.5, 0.5, "", transform=ax.transAxes, color='red', fontsize=10, ha='center', visible=False)

# Animation State
TRAILS_VISIBLE = True
pos_earth = initial_earth_pos(0)
pos_moon = initial_moon_pos(0)
pos_asteroid = initial_asteroid_pos(0, phase_solution)
vel_earth = EARTH_V0.copy()
vel_moon = MOON_V0.copy()
vel_asteroid = AST_V0.copy()
impact_displayed = False
is_paused = False
earth_velocity_arrow = None
asteroid_velocity_arrow = None

# Impact Details
def calculate_impact_coordinates(pos_e, pos_a, days_elapsed):
    rel_pos = (pos_a - pos_e) * AU_TO_M
    rot_angle = 2 * np.pi * days_elapsed / 365.25
    x_rot = rel_pos[0] * np.cos(rot_angle) + rel_pos[1] * np.sin(rot_angle)
    y_rot = -rel_pos[0] * np.sin(rot_angle) + rel_pos[1] * np.cos(rot_angle)
    z_rot = rel_pos[2]
    r = np.sqrt(x_rot**2 + y_rot**2 + z_rot**2)
    lat = np.degrees(np.arcsin(z_rot / r))
    lon = np.degrees(np.arctan2(y_rot, x_rot))
    lat_str = f"{abs(lat):.6f}° {'N' if lat >= 0 else 'S'}"
    lon_str = f"{abs(lon):.6f}° {'E' if lon >= 0 else 'W'}"
    return lat_str, lon_str

def show_impact_details(pos_e, pos_a, vel_a, days_elapsed):
    global impact_displayed
    impact_displayed = True
    v_impact = np.linalg.norm(vel_a) * AU_TO_M / DAY_TO_S
    ke = 0.5 * AST_MASS * v_impact**2
    crater_d = 1.161 * (AST_DENSITY / 2700)**0.33 * (ke / 1e6)**0.294 * (9.81 / v_impact**2)**0.147  # km
    lat_str, lon_str = calculate_impact_coordinates(pos_e, pos_a, days_elapsed)
    msg = (f"IMPACT!\nLat: {lat_str}, Lon: {lon_str}\n"
           f"Energy: {ke/4.184e12:.1f} Mt TNT\nCrater: {crater_d*1000:.1f} m")
    impact_text.set_text(msg)
    impact_text.set_visible(True)
    pygame.mixer.Sound("impact.wav").play()

# Animation Update
def update(frame):
    global impact_displayed, pos_earth, pos_moon, pos_asteroid, vel_earth, vel_moon, vel_asteroid
    if is_paused:
        return
    days_elapsed = t_days[frame]
    days_left = delta_days - days_elapsed
    current_date = start_date + datetime.timedelta(days=days_elapsed)

    pos_earth, pos_moon, pos_asteroid, vel_earth, vel_moon, vel_asteroid = update_positions_velocities(
        pos_earth, pos_moon, pos_asteroid, vel_earth, vel_moon, vel_asteroid, dt)

    # Force impact proximity check
    distance_EA = np.linalg.norm(pos_earth - pos_asteroid)
    if days_left <= 1 and distance_EA > 0.001:  # Within 1 day and not close enough
        pos_asteroid = pos_earth + (pos_asteroid - pos_earth) * 0.001 / distance_EA  # Nudge toward Earth

    if days_left <= 5:
        scale = max(0.1, days_left / 5 * 0.5)
        ax.set_xlim(pos_earth[0] - scale, pos_earth[0] + scale)
        ax.set_ylim(pos_earth[1] - scale, pos_earth[1] + scale)
        ax.set_zlim(pos_earth[2] - scale, pos_earth[2] + scale)

    if days_left <= 0 and not impact_displayed:
        show_impact_details(pos_earth, pos_asteroid, vel_asteroid, days_elapsed)
        ani.event_source.stop()

    if TRAILS_VISIBLE:
        earth_trail.append(pos_earth.copy())
        moon_trail.append(pos_moon.copy())
        asteroid_trail.append(pos_asteroid.copy())
        if len(earth_trail) > max_trail_length:
            earth_trail.pop(0)
            moon_trail.pop(0)
            asteroid_trail.pop(0)

    for line, arr in [(earth_trail_line, earth_trail), (moon_trail_line, moon_trail), (asteroid_trail_line, asteroid_trail)]:
        arr = np.array(arr) if arr else np.empty((0,3))
        if arr.size:
            line.set_data(arr[:,0], arr[:,1])
            line.set_3d_properties(arr[:,2])

    rot_angle = 2 * np.pi * days_elapsed / 1.0
    x_rot = earth_x_rot * np.cos(rot_angle) - earth_y_rot * np.sin(rot_angle)
    y_rot = earth_x_rot * np.sin(rot_angle) + earth_y_rot * np.cos(rot_angle)
    earth_surface.set_data((x_rot.flatten() * 0.01 + pos_earth[0]), (y_rot.flatten() * 0.01 + pos_earth[1]))
    earth_surface.set_3d_properties(earth_z_rot.flatten() * 0.01 + pos_earth[2])

    north.set_segments([[[pos_earth[0], pos_earth[1], pos_earth[2]], [pos_earth[0], pos_earth[1], pos_earth[2] + cardinal_scale]]])
    south.set_segments([[[pos_earth[0], pos_earth[1], pos_earth[2]], [pos_earth[0], pos_earth[1], pos_earth[2] - cardinal_scale]]])
    east.set_segments([[[pos_earth[0], pos_earth[1], pos_earth[2]], [pos_earth[0] + cardinal_scale, pos_earth[1], pos_earth[2]]]])
    west.set_segments([[[pos_earth[0], pos_earth[1], pos_earth[2]], [pos_earth[0] - cardinal_scale, pos_earth[1], pos_earth[2]]]])

    ke = 0.5 * AST_MASS * (np.linalg.norm(vel_asteroid) * AU_TO_M / DAY_TO_S)**2
    info_panel.set_text(f"Date: {current_date:%Y-%m-%d}\nDist: {distance_EA:.3f} AU")
    physics_panel.set_text(f"Mass: {AST_MASS:,.0f} kg\nKE: {ke/1e12:.1f} TJ")
    countdown_text.set_text(f"{days_left / 30.4375:.1f} months" if days_left > 0 else "IMPACT!")

    print(f"\rDate: {current_date:%Y-%m-%d} | Dist: {distance_EA:.3f} AU | "
          f"Vel_Ast: {np.linalg.norm(vel_asteroid):.3f} AU/day | KE: {ke/1e12:.1f} TJ", end='', flush=True)

    earth_marker._offsets3d = ([pos_earth[0]], [pos_earth[1]], [pos_earth[2]])
    moon_marker._offsets3d = ([pos_moon[0]], [pos_moon[1]], [pos_moon[2]])
    asteroid_marker._offsets3d = ([pos_asteroid[0]], [pos_asteroid[1]], [pos_asteroid[2]])
    asteroid_marker.set_color('red' if (frame // 5) % 2 == 0 else 'green')

    moon_orbit_shifted = np.array([pos_earth + np.array([R_MOON_ORBIT * np.cos(th), R_MOON_ORBIT * np.sin(th), 0])
                                   for th in theta_vals])
    moon_orbit_line.set_data(moon_orbit_shifted[:,0], moon_orbit_shifted[:,1])
    moon_orbit_line.set_3d_properties(moon_orbit_shifted[:,2])

    global earth_velocity_arrow, asteroid_velocity_arrow
    arrow_scale = 0.2
    if earth_velocity_arrow: earth_velocity_arrow.remove()
    earth_velocity_arrow = ax.quiver(pos_earth[0], pos_earth[1], pos_earth[2],
                                     vel_earth[0], vel_earth[1], vel_earth[2], color='cyan', length=arrow_scale)
    if asteroid_velocity_arrow: asteroid_velocity_arrow.remove()
    asteroid_velocity_arrow = ax.quiver(pos_asteroid[0], pos_asteroid[1], pos_asteroid[2],
                                        vel_asteroid[0], vel_asteroid[1], vel_asteroid[2], color='red', length=arrow_scale)

# Interactive Functions
def toggle_trails():
    global TRAILS_VISIBLE
    TRAILS_VISIBLE = not TRAILS_VISIBLE
    print(f"\nTrails: {TRAILS_VISIBLE}")

def deflect_asteroid(delta_v):
    global vel_asteroid
    vel_asteroid += np.array(delta_v)
    print(f"\nDeflection applied: {delta_v} AU/day")

def reset_simulation():
    global pos_earth, pos_moon, pos_asteroid, vel_earth, vel_moon, vel_asteroid, impact_displayed, earth_trail, moon_trail, asteroid_trail
    pos_earth = initial_earth_pos(0)
    pos_moon = initial_moon_pos(0)
    pos_asteroid = initial_asteroid_pos(0, phase_solution)
    vel_earth = EARTH_V0.copy()
    vel_moon = MOON_V0.copy()
    vel_asteroid = AST_V0.copy()
    impact_displayed = False
    impact_text.set_visible(False)
    earth_trail.clear()
    moon_trail.clear()
    asteroid_trail.clear()
    ax.set_xlim(-2, 2); ax.set_ylim(-2, 2); ax.set_zlim(-0.5, 0.5)
    print("\nSimulation reset")

def toggle_pause():
    global is_paused
    is_paused = not is_paused
    print(f"\nPaused: {is_paused}")

key_bindings = {
    't': toggle_trails,
    'd': lambda: deflect_asteroid([0.01, 0.01, 0.0]),
    'r': reset_simulation,
    'p': toggle_pause
}
fig.canvas.mpl_connect('key_press_event', lambda event: key_bindings.get(event.key, lambda: None)())

# Run Animation
pygame.mixer.init()
ani = FuncAnimation(fig, update, frames=FRAMES, interval=INTERVAL, blit=False)
handles = [mpl.lines.Line2D([], [], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Earth'),
           mpl.lines.Line2D([], [], marker='o', color='w', markerfacecolor='white', markersize=8, label='Moon'),
           mpl.lines.Line2D([], [], marker='o', color='w', markerfacecolor='green', markersize=8, label='Asteroid'),
           mpl.lines.Line2D([], [], marker='o', color='w', markerfacecolor='yellow', markersize=10, label='Sun')]
ax.legend(handles=handles, loc='upper right', fontsize='small')
plt.show()
