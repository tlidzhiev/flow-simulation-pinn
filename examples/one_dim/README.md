# Flow simulation

Consider the physics flow simulation of two liquids (oil and water) in porous environment, which is described by the following equations:

Conservation law: (differential form)

$$
\begin{cases}
\displaystyle \dfrac{\partial (\phi s_w \rho_w)}{\partial t} + \dfrac{\partial (\rho_w u_w)}{\partial \mathbf{x}} = 0 \\
\\
\displaystyle \dfrac{\partial (\phi s_o \rho_o)}{\partial t} + \dfrac{\partial (\rho_w u_w)}{\partial \mathbf{x}} = 0 \\
\end{cases}
$$

Conservation law: (integral form)

$$
\begin{cases}
\displaystyle \int\limits_{x_0}^{x_1} \left[ \phi s_w(\tau_1, \mathbf{x}) \rho_w(\tau_1, \mathbf{x}) - \phi s_w(\tau_0, \mathbf{x}) \rho_w(\tau_0, \mathbf{x}) \right] d\mathbf{x} \\
\quad + \int\limits_{\tau_0}^{\tau_1} \left[ \rho_w(t, x_1) u_w(t, x_1) - \rho_w(t, x_0) u_w(t, x_0) \right] dt = 0 \\
\\
\displaystyle \int\limits_{x_0}^{x_1} \left[ \phi s_o(\tau_1, \mathbf{x}) \rho_o(\tau_1, \mathbf{x}) - \phi s_o(\tau_0, \mathbf{x}) \rho_o(\tau_0, \mathbf{x}) \right] d\mathbf{x} \\
\quad + \int\limits_{\tau_0}^{\tau_1} \left[ \rho_o(t, x_1) u_o(t, x_1) - \rho_o(t, x_0) u_o(t, x_0) \right] dt = 0
\end{cases}
$$

Darcy's law:

$$
\begin{cases}
\displaystyle u_w = - \dfrac{k(\mathbf{x}) \, k_{r,w}(s_w)}{\mu_w} \dfrac{\partial p}{\partial \mathbf{x}} \\
\\
u_o = - \dfrac{k(\mathbf{x}) \, k_{r,o}(s_o)}{\mu_o} \dfrac{\partial p}{\partial \mathbf{x}}\\
\end{cases}
$$


where $s_w, s_o$ — water and oil saturation respectively, $p$ — pressure,
$u_w, u_o$ — water and oil flow velocity respectively,
$k_{r, w}(s) = k_{r, w}(s_w, s_o)$ и $k_{r, o}(s) = k_{r, o}(s_w, s_o)$ — relative phase permeabilities of water and oil respectively,
$\mu_w, \mu_o$ — viscosities of water and oil respectively

Initial condition:

$$
\begin{cases}
p(0, x) = 1 - x\\
s_w(0, x > 0) = 0\\
s_o(0, x > 0) = 1\\
\end{cases}
$$

Boundary condition:

$$
\begin{cases}
p(t, 0) = 1\\
p(t, 1) = 0\\
s_w(t, 0) = 1\\
s_o(t, 0) = 0\\
\end{cases}
$$
