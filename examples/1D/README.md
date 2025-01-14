# Flow simulation

Consider the physics flow simulation of two liquids (oil and water) in porous environment, which is described by the following equations:

Conservation law:

$$
\begin{dcases}
\dfrac{\partial}{\partial t} (\phi s_w \rho_w) + \sum\limits_{i = 1}^n \dfrac{\partial}{\partial x_i} (\rho_w u_{w, i}) = 0\\
%
\dfrac{\partial}{\partial t} (\phi s_o \rho_o) + \sum\limits_{i = 1}^n \dfrac{\partial}{\partial x_i} (\rho_o u_{o, i}) = 0\\
\end{dcases}
$$

Darcy's law:

$$
\begin{dcases}
u_w = - \dfrac{k(\mathbf{x}) k_{r, w}(s_w)}{\mu_w} \nabla_{\mathbf{x}} p(t, \mathbf{x})\\
u_o = - \dfrac{k(\mathbf{x}) k_{r, o}(s_o)}{\mu_o} \nabla_{\mathbf{x}}  p(t, \mathbf{x})\\
\end{dcases}
$$


where $s_w, s_o$ — water and oil saturation respectively, $p$ — pressure,
$u_w, u_o$ — water and oil flow velocity respectively,
$k_{r, w}(s) = k_{r, w}(s_w, s_o)$ и $k_{r, o}(s) = k_{r, o}(s_w, s_o)$ — relative phase permeabilities of water and oil respectively,
$\mu_w, \mu_o$ — viscosities of water and oil respectively

Initial condition:

$$
\begin{cases}
s_w(0, x) = 0\\
s_o(0, x) = 1\\
p(0, x) = 1 - x\\
\end{cases}
$$

Boundary condition:

$$
\begin{cases}
p(t, 0) = 1\\
p(t, 1) = 0\\
s_w(t, 0) = 1\\
s_o(t, 0) = 0\\
s_w(t, 1) = 0\\
s_o(t, 1) = 1\\
\end{cases}
$$
