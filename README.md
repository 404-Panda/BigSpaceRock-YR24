Asteroid Collision Simulator
The Asteroid Collision Simulator is a high-fidelity Python framework for modeling and visualizing the dynamics of an asteroid on a collision course with Earth. The project integrates advanced orbital mechanics, 3D coordinate transformations, and numerical optimization to produce an accurate, real-time simulation of celestial collisions.

Technical Overview
Orbital Mechanics and 3D Transformations
The simulator computes the positions of Earth, the Moon, and an asteroid in a heliocentric coordinate system. While Earth and Moon follow classical circular orbits, the asteroid is modeled using an elliptical trajectory defined by its semi-major axis 
𝑎
a and semi-minor axis 
𝑏
b.

The unrotated orbital positions of the asteroid are given by:

x_{\text{orb}}(t, \phi) = a \cos(\omega_{\text{ast}} t + \phi)

y_{\text{orb}}(t, \phi) = b \sin(\omega_{\text{ast}} t + \phi)

z_{\text{orb}}(t, \phi) = 0,
where 𝜔 ast = 2 𝜋 𝑇 ast ω  ast = T ast 2π
​and 
𝜙
ϕ is the phase parameter.

These positions are transformed into 3D space via two rotations:

Rotation about the x-axis (inclination 
𝑖 i):

R_x(i) = \begin{bmatrix} 1 & 0 & 0 \\ 0 & \cos i & -\sin i \\ 0 & \sin i & \cos i \end{bmatrix}
Rotation about the z-axis (longitude of the ascending node 
Ω  Ω):

R_z(\Omega) = \begin{bmatrix} \cos \Omega & -\sin \Omega & 0 \\ \sin \Omega & \cos \Omega & 0 \\ 0 & 0 & 1 \end{bmatrix}.

The final position is computed as:

\vec{r}_{\text{ast}} = R_z(\Omega) \cdot R_x(i) \cdot \begin{bmatrix} x_{\text{orb}} \\ y_{\text{orb}} \\ 0 \end{bmatrix}.
Collision Determination
The simulator first uses SciPy's fsolve to approximate the phase 
𝜙
ϕ that minimizes the function

f(\phi) = \left\| \vec{r}_{\text{Earth}}(T) - \vec{r}_{\text{ast}}(T, \phi) \right\|^2,
where 
𝑇
T is the impact time. To ensure that the collision occurs at an orbital crossing, the simulator forces the unrotated y-coordinate to be zero at impact, imposing:

\omega_{\text{ast}} T + \phi = n\pi \quad (n \in \{0,1\}).
Candidate phases are generated from this equation, and the one that minimizes the distance between Earth’s and the asteroid’s positions at 
𝑇
T is selected.

Impact Physics
Key impact parameters are computed using classical mechanics:

Kinetic Energy (KE):

KE = \frac{1}{2} m v^2,
Momentum (p):
p = m v.
Here, the asteroid's mass 
𝑚
m is calculated from its volume and density. A simplified atmospheric deceleration factor is applied to model energy loss during atmospheric entry.

3D Visualization and Animation
Rendering:
The simulation uses Matplotlib’s 3D plotting to render Earth, the Moon, and the asteroid alongside reference orbits.

Animation:
The FuncAnimation function updates positions, draws trails, and renders velocity vectors—computed via finite differences on stored trajectory data—at each simulation step.

Interactivity and Data Handling
Interactive Controls:
Functions such as toggle_trails(), simulate_deflection(delta_v_vector), and reset_simulation() allow users to interact with the simulation in real time.

Data Export:
The simulation logs position data along with timestamps and provides a CSV export functionality for further analysis.
