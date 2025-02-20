import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import datetime
import random
from scipy.optimize import fsolve
from mpl_toolkits.mplot3d import Axes3D  # Enable 3D plotting
import matplotlib as mpl
import csv  # For exporting simulation data

plt.style.use('dark_background')  # Dark

# =============================================================================
# 1. Simulation Dates and Time Setup
# =============================================================================
start_date = datetime.date(2025, 1, 1)
impact_date = datetime.date(2037, 12, 22)
delta_days = (impact_date - start_date).days

# Animation parameters
FRAMES = 600
INTERVAL = 40  # milliseconds between frames
t_days = np.linspace(0, delta_days, FRAMES)

# =============================================================================
# 2. Orbital Parameters & 3D Inclination Settings
# =============================================================================
PERIOD_EARTH = 365.25        # Earth orbit in days
PERIOD_MOON  = 27.3          # Moon orbit (approx)

# --- Change: Asteroid now has a 4-year orbit ---
PERIOD_ASTEROID = 4 * 365.25  # 4-year orbital period

omega_earth    = 2 * np.pi / PERIOD_EARTH
omega_moon     = 2 * np.pi / PERIOD_MOON
omega_asteroid = 2 * np.pi / PERIOD_ASTEROID

R_EARTH = 1.0                # Earth orbit radius in AU
R_MOON  = 0.00257            # Moon orbit radius in AU (scaled)

# --- Change: More elliptical asteroid orbit ---
a = 1.5  # semi-major axis (AU)
b = 0.5  # semi-minor axis (AU)

# Introduce an inclination for the asteroid's orbit in 3D (in radians)
AST_INCLINATION = np.radians(random.uniform(0, 10))  # small tilt (0-10°)
AST_ASC_NODE = np.radians(random.uniform(0, 360))      # longitude of ascending node

# =============================================================================
# 3. Define Essential Functions for Positions
# =============================================================================
def earth_position(t):
    """Earth's position on a circular orbit (z=0)."""
    xE = R_EARTH * np.cos(omega_earth * t)
    yE = R_EARTH * np.sin(omega_earth * t)
    zE = 0.0
    return np.array([xE, yE, zE])

def moon_position(t):
    """Moon's orbit around Earth (still in Earth's plane, z=0)."""
    xE, yE, _ = earth_position(t)
    scale_factor = PERIOD_EARTH / PERIOD_MOON
    xM_rel = R_MOON * np.cos(omega_moon * t * scale_factor)
    yM_rel = R_MOON * np.sin(omega_moon * t * scale_factor)
    zM_rel = 0.0
    return np.array([xE + xM_rel, yE + yM_rel, zM_rel])

def asteroid_position_given_phase(t, phase):
    """Compute asteroid position with a given phase and apply 3D rotations."""
    # Position in orbital plane (before rotation)
    x_orb = a * np.cos(omega_asteroid * t + phase)
    y_orb = b * np.sin(omega_asteroid * t + phase)
    z_orb = 0.0
    # Rotate about the x-axis by the inclination angle
    x_temp = x_orb
    y_temp = y_orb * np.cos(AST_INCLINATION) - z_orb * np.sin(AST_INCLINATION)
    z_temp = y_orb * np.sin(AST_INCLINATION) + z_orb * np.cos(AST_INCLINATION)
    # Rotate about the z-axis by the ascending node angle
    x_final = x_temp * np.cos(AST_ASC_NODE) - y_temp * np.sin(AST_ASC_NODE)
    y_final = x_temp * np.sin(AST_ASC_NODE) + y_temp * np.cos(AST_ASC_NODE)
    z_final = z_temp
    return np.array([x_final, y_final, z_final])

def asteroid_position(t):
    """Asteroid position using the solved phase_solution."""
    return asteroid_position_given_phase(t, phase_solution)

# =============================================================================
# 4. Determine the Collision Phase for a Precise Impact
# =============================================================================
T = delta_days
def collision_equation(phase):
    pos_e = earth_position(T)         # Earth's position at impact time (z=0)
    pos_a = asteroid_position_given_phase(T, phase)
    return np.linalg.norm(pos_e - pos_a)**2

# First, get an approximate solution (which may not satisfy the crossing condition)
phase_guess = 0.0
phase_solution = fsolve(collision_equation, phase_guess)[0]
candidate_n = [0, 1]
candidate_phases = [n * np.pi - omega_asteroid * T for n in candidate_n]
candidate_diffs = [np.linalg.norm(earth_position(T) - asteroid_position_given_phase(T, cp))
                   for cp in candidate_phases]
best_index = np.argmin(candidate_diffs)
phase_solution = candidate_phases[best_index]
# =============================================================================
# 5. Impact Physics: Asteroid Properties and Real Data
# =============================================================================
AST_DIAMETER_M = 40.0     # Diameter ~40 m
AST_SPEED_MS   = 17000.0  # Impact speed 17 km/s
AST_DENSITY    = 3000.0   # Typical stony density in kg/m³
AST_RADIUS_M   = AST_DIAMETER_M / 2
AST_VOLUME     = (4/3) * np.pi * (AST_RADIUS_M**3)
AST_MASS       = AST_DENSITY * AST_VOLUME
AST_KE_JOULES  = 0.5 * AST_MASS * (AST_SPEED_MS**2)

# Additional physics: Momentum and impact angle (assume near vertical entry)
AST_MOMENTUM   = AST_MASS * AST_SPEED_MS  # in kg·m/s
IMPACT_ANGLE   = random.uniform(60, 90)   # degrees from horizontal
IMPACT_ANGLE_RAD = np.radians(IMPACT_ANGLE)

# Atmospheric entry reduction (very simplified model)
ATMOSPHERIC_DECEL = 0.8  # 20% speed reduction due to atmosphere
AST_SPEED_AT_IMPACT = AST_SPEED_MS * ATMOSPHERIC_DECEL
AST_KE_AT_IMPACT = 0.5 * AST_MASS * (AST_SPEED_AT_IMPACT**2)

# =============================================================================
# 6. Create a 3D Starry Background
# =============================================================================
num_stars = 500
max_radius = 1.5 * a
star_r = max_radius * np.cbrt(np.random.rand(num_stars))
star_theta = 2 * np.pi * np.random.rand(num_stars)
star_phi = np.pi * np.random.rand(num_stars)  # random polar angle

star_x = star_r * np.sin(star_phi) * np.cos(star_theta)
star_y = star_r * np.sin(star_phi) * np.sin(star_theta)
star_z = star_r * np.cos(star_phi)

# =============================================================================
# 7. Figure and Axes Setup (3D)
# =============================================================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('black')
ax.grid(False)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(-0.5, 0.5)
ax.axis('off')

# Plot background stars
ax.scatter(star_x, star_y, star_z, s=1, c='white', alpha=0.7)

# Draw the Sun at the center (as a bright yellow sphere)
sun = ax.scatter(0, 0, 0, s=200, c='yellow', marker='o', label='Sun')

# =============================================================================
# 8. Plot Reference Orbits in 3D
# =============================================================================
orbit_points = 300
theta_vals = np.linspace(0, 2*np.pi, orbit_points)
# Earth orbit (z=0)
x_orbit_earth = R_EARTH * np.cos(theta_vals)
y_orbit_earth = R_EARTH * np.sin(theta_vals)
z_orbit_earth = np.zeros_like(theta_vals)
ax.plot(x_orbit_earth, y_orbit_earth, z_orbit_earth, c='blue', ls='--', alpha=0.5)

# Moon orbit (relative to Earth, then translated to Earth position)
x_orbit_moon = R_MOON * np.cos(theta_vals)
y_orbit_moon = R_MOON * np.sin(theta_vals)
z_orbit_moon = np.zeros_like(theta_vals)
moon_orbit_line, = ax.plot([], [], [], c='gray', ls='--', alpha=0.3)

# Asteroid orbit in 3D (for reference, without inclination effects)
x_orbit_ast = a * np.cos(theta_vals)
y_orbit_ast = b * np.sin(theta_vals)
z_orbit_ast = np.zeros_like(theta_vals)
x_ref = x_orbit_ast * np.cos(AST_ASC_NODE) - y_orbit_ast * np.sin(AST_ASC_NODE)
y_ref = x_orbit_ast * np.sin(AST_ASC_NODE) + y_orbit_ast * np.cos(AST_ASC_NODE)
z_ref = z_orbit_ast  # remains zero before inclination rotation
y_ref_tilt = y_ref * np.cos(AST_INCLINATION)
z_ref_tilt = y_ref * np.sin(AST_INCLINATION)
ax.plot(x_ref, y_ref_tilt, z_ref_tilt, c='green', ls='--', alpha=0.5)

# =============================================================================
# 9. Markers for Moving Objects
# =============================================================================
earth_marker = ax.scatter([], [], [], color='blue', s=50, label='Earth')
moon_marker  = ax.scatter([], [], [], color='white', s=20, label='Moon')
asteroid_marker = ax.scatter([], [], [], color='green', s=60, label='Asteroid')

# =============================================================================
# 10. Data Panels and Info Text
# =============================================================================
impact_info_text = ax.text2D(0.05, 0.92, "Asteroid 2024 YR4\nGuaranteed collision:\nDec 22, 2037",
                             transform=ax.transAxes, color='red', fontsize=9)

countdown_text = ax.text2D(0.5, 0.05, "", transform=ax.transAxes,
                           color='yellow', fontsize=10, fontweight='bold', ha='center')

info_panel = ax.text2D(0.02, 0.02, "", transform=ax.transAxes,
                       color='cyan', fontsize=8, va='bottom',
                       bbox=dict(facecolor='black', alpha=0.5, edgecolor='cyan'))

physics_panel = ax.text2D(0.75, 0.75, "", transform=ax.transAxes,
                          color='magenta', fontsize=8, ha='right',
                          bbox=dict(facecolor='black', alpha=0.5, edgecolor='magenta'))

# =============================================================================
# 11. Impact Details Setup
# =============================================================================
impact_displayed = False
random_lat = random.uniform(-90, 90)
random_lon = random.uniform(-180, 180)
crater_diameter_m = 20.0 * ((AST_KE_AT_IMPACT / 4e15)**(1/3))
casualty_estimate = random.randint(100000, 2000000)

impact_text = ax.text2D(0.5, 0.5, "", transform=ax.transAxes,
                        color='red', fontsize=10, fontweight='bold', ha='center',
                        va='center', visible=False)

def show_impact_details():
    global impact_displayed
    impact_displayed = True
    msg = (
        "IMPACT OCCURRED!\n\n"
        f"Kinetic Energy: {AST_KE_AT_IMPACT/4.184e9:,.1f} kilotons TNT\n"
        f"Momentum: {AST_MOMENTUM:,.2e} kg·m/s\n"
        f"Impact Angle: {IMPACT_ANGLE:,.1f}°\n"
        f"Crater ~{crater_diameter_m:,.1f} m\n"
        f"Location (lat, lon): ({random_lat:.1f}, {random_lon:.1f})\n"
        f"Estimated casualties: {casualty_estimate:,}\n"
        "Simulation End"
    )
    impact_text.set_text(msg)
    impact_text.set_visible(True)

# =============================================================================
# 12. Utility: Asteroid Color Flashing Function
# =============================================================================
def get_asteroid_color(frame):
    return 'red' if (frame // 5) % 2 == 0 else 'green'

# =============================================================================
# 13. Trajectory Trails Setup (Store positions for trailing effect)
# =============================================================================
earth_trail = []
moon_trail = []
asteroid_trail = []
max_trail_length = 50

earth_trail_line, = ax.plot([], [], [], 'b-', lw=1, alpha=0.5)
moon_trail_line, = ax.plot([], [], [], 'w-', lw=1, alpha=0.5)
asteroid_trail_line, = ax.plot([], [], [], 'g-', lw=1, alpha=0.5)

# =============================================================================
# New Global Variables for Additional Features
# =============================================================================
TRAILS_VISIBLE = True
ASTEROID_DEFLECTION = np.array([0.0, 0.0, 0.0])
simulation_data = []  # To store simulation data for export
# Variables to hold velocity arrow artists
earth_velocity_arrow = None
moon_velocity_arrow = None
asteroid_velocity_arrow = None

# =============================================================================
# 14. Animation Update Function (3D, with physics, trails, velocity vectors, and additional features)
# =============================================================================
def update(frame):
    global impact_displayed, earth_trail, moon_trail, asteroid_trail
    global earth_velocity_arrow, moon_velocity_arrow, asteroid_velocity_arrow, simulation_data
    days_elapsed = t_days[frame]
    days_left = delta_days - days_elapsed
    current_date = start_date + datetime.timedelta(days=days_elapsed)

    pos_earth = earth_position(days_elapsed)
    pos_moon = moon_position(days_elapsed)
    pos_asteroid = asteroid_position(days_elapsed)
    
    # New Feature: Apply deflection offset to asteroid position
    pos_asteroid = pos_asteroid + ASTEROID_DEFLECTION

    # Force collision at impact time:
    if days_left <= 0:
        pos_asteroid = pos_earth.copy()

    # Update simulation data storage
    simulation_data.append((current_date, pos_earth, pos_moon, pos_asteroid))

    # Update trails only if trails are enabled
    if TRAILS_VISIBLE:
        earth_trail.append(pos_earth)
        moon_trail.append(pos_moon)
        asteroid_trail.append(pos_asteroid)
        if len(earth_trail) > max_trail_length:
            earth_trail.pop(0)
            moon_trail.pop(0)
            asteroid_trail.pop(0)
    else:
        # If trails are toggled off, clear the data and lines
        earth_trail.clear()
        moon_trail.clear()
        asteroid_trail.clear()
        earth_trail_line.set_data([], [])
        earth_trail_line.set_3d_properties([])
        moon_trail_line.set_data([], [])
        moon_trail_line.set_3d_properties([])
        asteroid_trail_line.set_data([], [])
        asteroid_trail_line.set_3d_properties([])

    earth_trail_arr = np.array(earth_trail) if earth_trail else np.empty((0,3))
    moon_trail_arr = np.array(moon_trail) if moon_trail else np.empty((0,3))
    asteroid_trail_arr = np.array(asteroid_trail) if asteroid_trail else np.empty((0,3))
    if earth_trail_arr.size:
        earth_trail_line.set_data(earth_trail_arr[:,0], earth_trail_arr[:,1])
        earth_trail_line.set_3d_properties(earth_trail_arr[:,2])
    if moon_trail_arr.size:
        moon_trail_line.set_data(moon_trail_arr[:,0], moon_trail_arr[:,1])
        moon_trail_line.set_3d_properties(moon_trail_arr[:,2])
    if asteroid_trail_arr.size:
        asteroid_trail_line.set_data(asteroid_trail_arr[:,0], asteroid_trail_arr[:,1])
        asteroid_trail_line.set_3d_properties(asteroid_trail_arr[:,2])

    distance_EA = np.linalg.norm(pos_earth - pos_asteroid)
    info_panel.set_text(
        f"Sim Date: {current_date.strftime('%Y-%m-%d')}\n"
        f"Earth-Asteroid Dist: {distance_EA:0.3f} AU\n"
        f"Asteroid KE: {AST_KE_AT_IMPACT/1e12:,.2f} TJ"
    )
    physics_panel.set_text(
        f"Asteroid Mass: {AST_MASS:,.2f} kg\n"
        f"Momentum: {AST_MOMENTUM:,.2e} kg·m/s\n"
        f"Impact Angle: {IMPACT_ANGLE:,.1f}°\n"
        f"Speed at Impact: {AST_SPEED_AT_IMPACT:,.1f} m/s"
    )

    if days_left <= 0 and not impact_displayed:
        countdown_text.set_text("IMPACT!")
        show_impact_details()
        ani.event_source.stop()
    elif days_left <= 0 and impact_displayed:
        countdown_text.set_text("IMPACT!")
    else:
        months_left = days_left / 30.4375
        countdown_text.set_text(f"{months_left:0.1f} months to impact")

    earth_marker._offsets3d = ([pos_earth[0]], [pos_earth[1]], [pos_earth[2]])
    moon_marker._offsets3d = ([pos_moon[0]], [pos_moon[1]], [pos_moon[2]])
    asteroid_marker._offsets3d = ([pos_asteroid[0]], [pos_asteroid[1]], [pos_asteroid[2]])
    asteroid_marker.set_color(get_asteroid_color(frame))

    moon_orbit_shifted = np.array([pos_earth + np.array([R_MOON * np.cos(th), R_MOON * np.sin(th), 0])
                                   for th in theta_vals])
    moon_orbit_line.set_data(moon_orbit_shifted[:,0], moon_orbit_shifted[:,1])
    moon_orbit_line.set_3d_properties(moon_orbit_shifted[:,2])
    
    # New Feature: Calculate and display velocity vectors (using finite differences)
    global earth_velocity_arrow, moon_velocity_arrow, asteroid_velocity_arrow
    arrow_scale = 0.2  # scaling factor for arrow length
    if len(earth_trail) >= 2:
        v_earth = earth_trail[-1] - earth_trail[-2]
        if earth_velocity_arrow is not None:
            earth_velocity_arrow.remove()
        earth_velocity_arrow = ax.quiver(pos_earth[0], pos_earth[1], pos_earth[2],
                                         v_earth[0], v_earth[1], v_earth[2],
                                         color='cyan', length=arrow_scale, normalize=True)
    if len(moon_trail) >= 2:
        v_moon = moon_trail[-1] - moon_trail[-2]
        if moon_velocity_arrow is not None:
            moon_velocity_arrow.remove()
        moon_velocity_arrow = ax.quiver(pos_moon[0], pos_moon[1], pos_moon[2],
                                        v_moon[0], v_moon[1], v_moon[2],
                                        color='magenta', length=arrow_scale, normalize=True)
    if len(asteroid_trail) >= 2:
        v_asteroid = asteroid_trail[-1] - asteroid_trail[-2]
        if asteroid_velocity_arrow is not None:
            asteroid_velocity_arrow.remove()
        asteroid_velocity_arrow = ax.quiver(pos_asteroid[0], pos_asteroid[1], pos_asteroid[2],
                                            v_asteroid[0], v_asteroid[1], v_asteroid[2],
                                            color='red', length=arrow_scale, normalize=True)

    return (earth_marker, moon_marker, asteroid_marker, countdown_text,
            impact_text, info_panel, physics_panel,
            earth_trail_line, moon_trail_line, asteroid_trail_line, moon_orbit_line,
            earth_velocity_arrow, moon_velocity_arrow, asteroid_velocity_arrow)

# =============================================================================
# 15. Run the Animation
# =============================================================================
ani = FuncAnimation(
    fig,
    update,
    frames=FRAMES,
    interval=INTERVAL,
    blit=False
)

handles = [mpl.lines.Line2D([], [], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Earth'),
           mpl.lines.Line2D([], [], marker='o', color='w', markerfacecolor='white', markersize=8, label='Moon'),
           mpl.lines.Line2D([], [], marker='o', color='w', markerfacecolor='green', markersize=8, label='Asteroid'),
           mpl.lines.Line2D([], [], marker='o', color='w', markerfacecolor='yellow', markersize=10, label='Sun')]
ax.legend(handles=handles, loc='upper right', fontsize='small')

# =============================================================================
# New Functions for Additional Features
# =============================================================================
def toggle_trails():
    """Toggle the visibility of trajectory trails."""
    global TRAILS_VISIBLE, earth_trail, moon_trail, asteroid_trail
    TRAILS_VISIBLE = not TRAILS_VISIBLE
    if not TRAILS_VISIBLE:
        earth_trail.clear()
        moon_trail.clear()
        asteroid_trail.clear()
        earth_trail_line.set_data([], [])
        earth_trail_line.set_3d_properties([])
        moon_trail_line.set_data([], [])
        moon_trail_line.set_3d_properties([])
        asteroid_trail_line.set_data([], [])
        asteroid_trail_line.set_3d_properties([])
    print(f"Trails visible: {TRAILS_VISIBLE}")

def simulate_deflection(delta_v_vector):
    """
    Simulate a deflection maneuver by applying a velocity change (position offset)
    to the asteroid's trajectory.
    
    Parameters:
        delta_v_vector (iterable): A 3-element vector to add to the asteroid's position.
    """
    global ASTEROID_DEFLECTION
    ASTEROID_DEFLECTION = np.array(delta_v_vector)
    print(f"Asteroid deflection applied: {ASTEROID_DEFLECTION}")

def export_simulation_data(filename="simulation_data.csv"):
    """
    Export the recorded simulation data (date and positions of Earth, Moon, and Asteroid)
    to a CSV file.
    """
    global simulation_data
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Date", "Earth_x", "Earth_y", "Earth_z",
                         "Moon_x", "Moon_y", "Moon_z",
                         "Asteroid_x", "Asteroid_y", "Asteroid_z"])
        for data in simulation_data:
            date, pos_e, pos_m, pos_a = data
            writer.writerow([date.strftime("%Y-%m-%d"),
                             pos_e[0], pos_e[1], pos_e[2],
                             pos_m[0], pos_m[1], pos_m[2],
                             pos_a[0], pos_a[1], pos_a[2]])
    print(f"Simulation data exported to {filename}")

def reset_simulation():
    """Reset the simulation trails and recorded data."""
    global earth_trail, moon_trail, asteroid_trail, simulation_data
    earth_trail.clear()
    moon_trail.clear()
    asteroid_trail.clear()
    simulation_data.clear()
    print("Simulation reset.")

# =============================================================================
# End of Script
# =============================================================================
plt.show()
