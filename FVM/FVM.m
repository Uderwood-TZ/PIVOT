function spherical_gradient_chaos_fvm


clc;
close all;

%% 1. Output directory
rootDir = fileparts(mfilename('fullpath'));
if isempty(rootDir)
    rootDir = pwd;
end
outDir = fullfile(rootDir, 'spherical_gradient_results');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

%% 2. Physical parameters (SI units)
p.a       = 1.00e-2;       % reference inner radius [m]
p.b       = 2.00e-2;       % reference outer radius [m]
p.rho0    = 1050.0;        % reference density [kg/m^3]
p.mu      = 2.00e5;        % shear modulus [Pa]
p.kappa   = 8.00e5;        % volumetric modulus parameter [Pa]
p.ell_g   = 6.00e-4;       % intrinsic strain-gradient length [m]
p.g       = p.mu*p.ell_g^2;% strain-gradient coefficient [Pa m^2]
p.eta     = 35.0;          % shear-type viscosity [Pa s]
p.zeta    = 80.0;          % volumetric viscosity [Pa s]
p.gamma_g = 1.40;          % adiabatic exponent [-]
p.p0      = 101325.0;      % common equilibrium absolute pressure [Pa]

% Periodic external gauge-pressure forcing:
p.DeltaP  = 0.12*p.mu;     % forcing amplitude [Pa]
p.Omega   = 9.00e3;        % angular forcing frequency [rad/s]
p.nRamp   = 4.0;           % smooth ramp duration measured in forcing periods
p.nPeriod = 30.0;          % total simulated forcing periods

%% 3. Spatial and temporal grids
p.N  = 121;                % radial nodes
p.r  = linspace(p.a, p.b, p.N).';
p.dr = p.r(2)-p.r(1);

% Cell faces for conservative spherical finite-volume balance.
p.rf = zeros(p.N+1,1);
p.rf(1)       = p.a;
p.rf(end)     = p.b;
p.rf(2:p.N)   = 0.5*(p.r(1:end-1)+p.r(2:end));

Tforce   = 2*pi/p.Omega;
p.tauRamp = p.nRamp*Tforce;
tEnd     = p.nPeriod*Tforce;
NtOut    = 1201;
tOut     = linspace(0.0, tEnd, NtOut).';

% Numerical safeguarding is only for rejected internal Newton/BDF trial states.
% A post-processing assertion checks that the accepted solution remains far
% above this floor, so the final reported solution does not rely on clipping.
p.lambdaFloor = 5.0e-2;

%% 4. Initial conditions
u0 = zeros(p.N,1);
v0 = zeros(p.N,1);
y0 = [u0; v0];

%% 5. Time integration
fprintf('Running nonlinear spherical strain-gradient simulation...\n');
fprintf('N = %d radial nodes, output times = %d\n', p.N, NtOut);
fprintf('Forcing period = %.6e s, t_end = %.6e s\n', Tforce, tEnd);

opts = odeset( ...
    'RelTol', 2.0e-6, ...
    'AbsTol', 1.0e-9, ...
    'MaxStep', Tforce/50.0, ...
    'InitialStep', Tforce/500.0);

[t, Y] = ode15s(@(tt,yy) rhs_system(tt,yy,p), tOut, y0, opts);

U = Y(:,1:p.N);
V = Y(:,p.N+1:2*p.N);

%% 6. Post-processing all local fields
fprintf('Post-processing all local r-t fields...\n');

fieldNames = { ...
    'u', 'v', 'acceleration', 'current_radius', ...
    'lambda_r', 'lambda_theta', 'green_strain_r', 'green_strain_theta', 'hencky_strain_r', 'hencky_strain_theta', 'J', 'logJ', 'logJ_t', 'logJ_r', ...
    'q_gradient', 'f_gradient', ...
    'P_r_elastic', 'P_theta_elastic', ...
    'P_r_viscous', 'P_theta_viscous', ...
    'P_r_local', 'P_theta_local', ...
    'P_r_gradient', 'P_theta_gradient', ...
    'S_r_generalized', 'S_theta_generalized', ...
    'sigma_rr_local', 'sigma_tt_local', 'delta_sigma_local', 'sigma_mean_local', 'sigma_von_mises_local', ...
    'kinetic_energy_density', 'elastic_energy_density', ...
    'gradient_energy_density', 'stored_energy_density', ...
    'dissipation_density'};

fieldLabels = { ...
    'Radial displacement u', 'Radial velocity v', 'Radial acceleration a_r', 'Current radius x=r+u', ...
    'Radial stretch lambda_r', 'Hoop stretch lambda_theta', 'Radial Green-Lagrange strain', 'Hoop Green-Lagrange strain', 'Radial Hencky strain', 'Hoop Hencky strain', 'Jacobian J', 'Volumetric strain log(J)', 'Volumetric strain rate d(logJ)/dt', 'Radial gradient d(logJ)/dr', ...
    'Gradient microstress q', 'Strain-gradient force density f_g', ...
    'Elastic radial Piola stress', 'Elastic hoop Piola stress', ...
    'Viscous radial Piola stress', 'Viscous hoop Piola stress', ...
    'Local radial Piola stress', 'Local hoop Piola stress', ...
    'Gradient radial nominal stress', 'Gradient hoop nominal stress', ...
    'Generalized radial nominal stress', 'Generalized hoop nominal stress', ...
    'Local radial Cauchy stress', 'Local hoop Cauchy stress', 'Local hoop-radial Cauchy stress difference', 'Local mean Cauchy stress', 'Local von Mises stress', ...
    'Kinetic energy density', 'Hyperelastic energy density', ...
    'Gradient energy density', 'Total stored energy density', ...
    'Viscous dissipation density'};

fieldUnits = { ...
    'm', 'm/s', 'm/s^2', 'm', ...
    '1', '1', '1', '1', '1', '1', '1', '1', '1/s', '1/m', ...
    'Pa m', 'N/m^3', ...
    'Pa', 'Pa', ...
    'Pa', 'Pa', ...
    'Pa', 'Pa', ...
    'Pa', 'Pa', ...
    'Pa', 'Pa', ...
    'Pa', 'Pa', 'Pa', 'Pa', 'Pa', ...
    'J/m^3', 'J/m^3', ...
    'J/m^3', 'J/m^3', ...
    'W/m^3'};

nFields = numel(fieldNames);
Fmat = cell(nFields,1);
for j = 1:nFields
    Fmat{j} = zeros(numel(t), p.N);
end

% Scalar histories and global integrals.
Rc       = zeros(numel(t),1);
Ro       = zeros(numel(t),1);
Vc       = zeros(numel(t),1);
pcAbs    = zeros(numel(t),1);
pcGauge  = zeros(numel(t),1);
poGauge  = zeros(numel(t),1);
Ekin     = zeros(numel(t),1);
Eelastic = zeros(numel(t),1);
Egrad    = zeros(numel(t),1);
Estored  = zeros(numel(t),1);
Emech    = zeros(numel(t),1);
Ddot     = zeros(numel(t),1);
minJ     = zeros(numel(t),1);
maxJ     = zeros(numel(t),1);
maxAbsU  = zeros(numel(t),1);
maxAbsV  = zeros(numel(t),1);
maxAbsQ  = zeros(numel(t),1);
maxAbsFg = zeros(numel(t),1);
maxAbsSrr = zeros(numel(t),1);
maxAbsStt = zeros(numel(t),1);

for k = 1:numel(t)
    uk = U(k,:).';
    vk = V(k,:).';
    L  = local_fields(uk, vk, p, false);
    dY = rhs_system(t(k), [uk;vk], p);
    ak = dY(p.N+1:end);

    values = { ...
        uk, vk, ak, p.r+uk, ...
        L.lambda_r, L.lambda_t, L.Egreen_r, L.Egreen_t, L.hencky_r, L.hencky_t, L.J, L.e, L.edot, L.er, ...
        L.q, L.fg, ...
        L.Pr_e, L.Pt_e, ...
        L.Pr_v, L.Pt_v, ...
        L.Pr, L.Pt, ...
        L.Pr_g, L.Pt_g, ...
        L.Sr, L.St, ...
        L.sigma_rr, L.sigma_tt, L.delta_sigma, L.sigma_mean, L.sigma_vm, ...
        L.K, L.We, ...
        L.Wg, L.Wstored, ...
        L.D};

    for j = 1:nFields
        Fmat{j}(k,:) = values{j}.';
    end

    [pcAbs(k), pcGauge(k), poGauge(k), Rc(k), Ro(k), Vc(k)] = pressure_state(t(k), uk, p);

    Ekin(k)     = 4*pi*trapz(p.r, L.K      .*p.r.^2);
    Eelastic(k) = 4*pi*trapz(p.r, L.We     .*p.r.^2);
    Egrad(k)    = 4*pi*trapz(p.r, L.Wg     .*p.r.^2);
    Estored(k)  = Eelastic(k)+Egrad(k);
    Emech(k)    = Ekin(k)+Estored(k);
    Ddot(k)     = 4*pi*trapz(p.r, L.D      .*p.r.^2);

    minJ(k)      = min(L.J);
    maxJ(k)      = max(L.J);
    maxAbsU(k)   = max(abs(uk));
    maxAbsV(k)   = max(abs(vk));
    maxAbsQ(k)   = max(abs(L.q));
    maxAbsFg(k)  = max(abs(L.fg));
    maxAbsSrr(k) = max(abs(L.sigma_rr));
    maxAbsStt(k) = max(abs(L.sigma_tt));
end

% Strict physical admissibility check for the accepted trajectory.
tmpLR = Fmat{strcmp(fieldNames,'lambda_r')};
tmpLT = Fmat{strcmp(fieldNames,'lambda_theta')};
tmpJ  = Fmat{strcmp(fieldNames,'J')};
minLambdaR = min(tmpLR(:));
minLambdaT = min(tmpLT(:));
minJAll    = min(tmpJ(:));
if minLambdaR <= 10*p.lambdaFloor || minLambdaT <= 10*p.lambdaFloor || minJAll <= (10*p.lambdaFloor)^3
    warning(['Accepted trajectory approaches the numerical positivity floor. ', ...
             'Reduce DeltaP and/or refine the grid before using the result quantitatively.']);
end

%% 7. Save all local r-t field figures and triplet TXT files
fprintf('Saving %d local-field figures and TXT data files...\n', nFields);
for j = 1:nFields
    baseName = sprintf('field_%02d_%s', j, fieldNames{j});
    save_rt_field(outDir, baseName, p.r, t, Fmat{j}, fieldLabels{j}, fieldUnits{j});
end

%% 8. Scalar time histories
% Boundary histories extracted from field arrays.
uInner = U(:,1);
uOuter = U(:,end);
vInner = V(:,1);
vOuter = V(:,end);
aField = Fmat{strcmp(fieldNames,'acceleration')};
aInner = aField(:,1);
aOuter = aField(:,end);

% Cumulative viscous dissipation by trapezoidal integration.
EcumDiss = cumtrapz(t, Ddot);

curveNames = { ...
    'inner_radius', 'outer_radius', 'cavity_volume', ...
    'cavity_pressure_absolute', 'cavity_pressure_gauge', 'outer_pressure_gauge', ...
    'u_inner', 'u_outer', 'v_inner', 'v_outer', 'a_inner', 'a_outer', ...
    'kinetic_energy', 'elastic_energy', 'gradient_energy', 'stored_energy', 'mechanical_energy', ...
    'dissipation_rate', 'cumulative_dissipation', ...
    'min_J', 'max_J', 'max_abs_u', 'max_abs_v', 'max_abs_q', 'max_abs_f_gradient', ...
    'max_abs_sigma_rr', 'max_abs_sigma_tt'};

curveLabels = { ...
    'Inner current radius', 'Outer current radius', 'Cavity volume', ...
    'Cavity absolute pressure', 'Cavity gauge pressure', 'Applied outer gauge pressure', ...
    'Inner-surface displacement', 'Outer-surface displacement', 'Inner-surface velocity', 'Outer-surface velocity', 'Inner-surface acceleration', 'Outer-surface acceleration', ...
    'Total kinetic energy', 'Total hyperelastic energy', 'Total gradient energy', 'Total stored energy', 'Total mechanical energy', ...
    'Total viscous dissipation rate', 'Cumulative viscous dissipation', ...
    'Minimum J in solid', 'Maximum J in solid', 'Maximum |u|', 'Maximum |v|', 'Maximum |q|', 'Maximum |f_g|', ...
    'Maximum |sigma_rr|', 'Maximum |sigma_theta_theta|'};

curveUnits = { ...
    'm', 'm', 'm^3', ...
    'Pa', 'Pa', 'Pa', ...
    'm', 'm', 'm/s', 'm/s', 'm/s^2', 'm/s^2', ...
    'J', 'J', 'J', 'J', 'J', ...
    'W', 'J', ...
    '1', '1', 'm', 'm/s', 'Pa m', 'N/m^3', ...
    'Pa', 'Pa'};

curveValues = { ...
    Rc, Ro, Vc, ...
    pcAbs, pcGauge, poGauge, ...
    uInner, uOuter, vInner, vOuter, aInner, aOuter, ...
    Ekin, Eelastic, Egrad, Estored, Emech, ...
    Ddot, EcumDiss, ...
    minJ, maxJ, maxAbsU, maxAbsV, maxAbsQ, maxAbsFg, ...
    maxAbsSrr, maxAbsStt};

fprintf('Saving %d scalar time-history figures and TXT data files...\n', numel(curveNames));
for j = 1:numel(curveNames)
    baseName = sprintf('curve_%02d_%s', j, curveNames{j});
    save_time_curve(outDir, baseName, t, curveValues{j}, curveLabels{j}, curveUnits{j});
end

%% 9. Nonlinear-dynamics diagnostics from the outer-surface response
% These do not replace the requested t-curves; they provide additional CSF-style
% diagnostics from the same simulation. Every diagnostic also gets its own TXT.
transientPeriods = max(p.nRamp+2.0, 0.35*p.nPeriod);
tKeep = transientPeriods*Tforce;
keep = t >= tKeep;

% Phase portrait: TXT columns are u_outer, v_outer, time.
save_phase_portrait(outDir, 'diagnostic_phase_outer', uOuter(keep), vOuter(keep), t(keep));

% FFT amplitude spectrum after transient removal.
save_fft_spectrum(outDir, 'diagnostic_fft_outer', t(keep), uOuter(keep));

% Stroboscopic Poincare samples at one forcing period.
save_poincare(outDir, 'diagnostic_poincare_outer', t, uOuter, vOuter, Tforce, tKeep);

%% 10. Save aggregate MAT file and parameter summary
save(fullfile(outDir,'all_results.mat'), ...
    'p','t','U','V','fieldNames','fieldLabels','fieldUnits','Fmat', ...
    'curveNames','curveLabels','curveUnits','curveValues','-v7.3');
write_parameter_summary(fullfile(outDir,'parameter_summary.txt'), p, Tforce, tEnd, NtOut, ...
    minLambdaR, minLambdaT, minJAll);

fprintf('\nCompleted.\n');
fprintf('All figures and TXT data are in:\n%s\n', outDir);

end

%% ========================================================================
function dy = rhs_system(t, y, p)
% Semi-discrete governing equations.
N = p.N;
u = y(1:N);
v = y(N+1:2*N);

L = local_fields(u, v, p, true);
[~, pcGauge, poGauge] = pressure_state(t, u, p);

% Generalized radial traction at actual material boundaries. Because q=0 at
% both boundaries, generalized traction reduces to Sr=P_r+P_r^gradient there.
% Pressure is defined on the current spherical area; nominal traction therefore
% contains lambda_theta^2.
Sface = zeros(N+1,1);
Sface(1)   = -pcGauge * L.lambda_t(1)^2;
Sface(end) = -poGauge * L.lambda_t(end)^2;

% Interior face traction: centered second-order reconstruction.
Sface(2:N) = 0.5*(L.Sr(1:end-1)+L.Sr(2:end));

acc = zeros(N,1);
for i = 1:N
    rm = p.rf(i);
    rp = p.rf(i+1);
    Vi = (rp^3-rm^3)/3.0;  % integral r^2 dr over the spherical control volume

    radialFlux = rp^2*Sface(i+1) - rm^2*Sface(i);
    hoopTerm   = 2.0*p.r(i)*L.St(i)*(rp-rm);

    acc(i) = (radialFlux-hoopTerm)/(p.rho0*Vi);
end

dy = [v; acc];
end

%% ========================================================================
function L = local_fields(u, v, p, useSafeguard)
% Constitutive, strain-gradient, stress, energy, and dissipation fields.

ur = d1_second_order(u, p.dr);
vr = d1_second_order(v, p.dr);

lambda_r_raw = 1.0 + ur;
lambda_t_raw = 1.0 + u./p.r;

if useSafeguard
    % Safeguard only rejected implicit-solver trial states. The accepted
    % trajectory is checked afterward and must remain well above this floor.
    lambda_r = max(lambda_r_raw, p.lambdaFloor);
    lambda_t = max(lambda_t_raw, p.lambdaFloor);
else
    lambda_r = lambda_r_raw;
    lambda_t = lambda_t_raw;
end

J = lambda_r.*lambda_t.^2;
if useSafeguard
    J = max(J, p.lambdaFloor^3);
end

e = log(J);
edot = vr./lambda_r + 2.0*v./(p.r+u);

% Natural higher-order boundary condition q=g*e_r=0 at r=a,b.
er = d1_second_order(e, p.dr);
er(1)   = 0.0;
er(end) = 0.0;
q = p.g*er;

% M = d(r^2 q)/dr.
M = d1_second_order((p.r.^2).*q, p.dr);

% Compressible Neo-Hookean local first-Piola principal stresses.
Pr_e = p.mu*(lambda_r-1.0./lambda_r) + p.kappa*e./lambda_r;
Pt_e = p.mu*(lambda_t-1.0./lambda_t) + p.kappa*e./lambda_t;

% Kelvin-Voigt-type finite-deformation dissipation stresses.
Pr_v = p.eta*vr + p.zeta*edot./lambda_r;
Pt_v = p.eta*(v./p.r) + p.zeta*edot./lambda_t;

Pr = Pr_e + Pr_v;
Pt = Pt_e + Pt_v;

% Gradient contribution rewritten as generalized nominal stresses so that
% div(S_gradient) equals the variational strain-gradient force density.
Pr_g = -M./(lambda_r.*p.r.^2);
Pt_g = -M./(p.r.*(p.r+u));

Sr = Pr + Pr_g;
St = Pt + Pt_g;

% Explicit gradient force density for diagnostics.
fg = d1_second_order(Pr_g, p.dr) + 2.0*(Pr_g-Pt_g)./p.r;

% Local Cauchy stresses associated with the local Piola part.
sigma_rr = (lambda_r./J).*Pr;
sigma_tt = (lambda_t./J).*Pt;
delta_sigma = sigma_tt-sigma_rr;
sigma_mean = (sigma_rr+2.0*sigma_tt)/3.0;
sigma_vm = abs(sigma_rr-sigma_tt);

% Finite-strain measures.
Egreen_r = 0.5*(lambda_r.^2-1.0);
Egreen_t = 0.5*(lambda_t.^2-1.0);
hencky_r = log(lambda_r);
hencky_t = log(lambda_t);

% Energies and viscous dissipation.
We = 0.5*p.mu*(lambda_r.^2 + 2.0*lambda_t.^2 - 3.0 - 2.0*e) ...
     +0.5*p.kappa*e.^2;
Wg = 0.5*p.g*er.^2;
Wstored = We+Wg;
K  = 0.5*p.rho0*v.^2;
D  = p.eta*(vr.^2 + 2.0*(v./p.r).^2) + p.zeta*edot.^2;

L.lambda_r = lambda_r;
L.lambda_t = lambda_t;
L.Egreen_r = Egreen_r;
L.Egreen_t = Egreen_t;
L.hencky_r = hencky_r;
L.hencky_t = hencky_t;
L.J = J;
L.e = e;
L.edot = edot;
L.er = er;
L.q = q;
L.M = M;
L.Pr_e = Pr_e;
L.Pt_e = Pt_e;
L.Pr_v = Pr_v;
L.Pt_v = Pt_v;
L.Pr = Pr;
L.Pt = Pt;
L.Pr_g = Pr_g;
L.Pt_g = Pt_g;
L.Sr = Sr;
L.St = St;
L.fg = fg;
L.sigma_rr = sigma_rr;
L.sigma_tt = sigma_tt;
L.delta_sigma = delta_sigma;
L.sigma_mean = sigma_mean;
L.sigma_vm = sigma_vm;
L.We = We;
L.Wg = Wg;
L.Wstored = Wstored;
L.K = K;
L.D = D;
end

%% ========================================================================
function [pcAbs, pcGauge, poGauge, Rc, Ro, Vc] = pressure_state(t, u, p)
% Nonlinear adiabatic cavity pressure and smooth periodic external pressure.
Rc = p.a + u(1);
Ro = p.b + u(end);

% Keep rejected implicit trial states finite. The accepted trajectory must
% remain physical; post-processing verifies the deformation separately.
RcSafe = max(Rc, 0.05*p.a);
Vc = (4*pi/3.0)*RcSafe^3;
V0 = (4*pi/3.0)*p.a^3;

pcAbs   = p.p0*(V0/Vc)^p.gamma_g;
pcGauge = pcAbs-p.p0;

S = 1.0-exp(-(t/p.tauRamp)^2);
poGauge = p.DeltaP*S*sin(p.Omega*t);
end

%% ========================================================================
function df = d1_second_order(f, dr)
% Second-order finite-difference first derivative on a uniform grid.
N = numel(f);
if N < 3
    error('At least three radial nodes are required.');
end

df = zeros(size(f));
df(2:N-1) = (f(3:N)-f(1:N-2))/(2.0*dr);
df(1)     = (-3.0*f(1)+4.0*f(2)-f(3))/(2.0*dr);
df(N)     = ( 3.0*f(N)-4.0*f(N-1)+f(N-2))/(2.0*dr);
end

%% ========================================================================
function save_rt_field(outDir, baseName, r, t, Z, quantityLabel, unitLabel)
% Save one r-t color map and one three-column r,t,value TXT file.
fig = figure('Visible','off','Color','w','Position',[100 100 980 700]);
imagesc(r, t, Z);
axis xy;
axis tight;
colormap(jet(256));
cb = colorbar;
cb.Label.String = unitLabel;
xlabel('r (m)','Interpreter','none');
ylabel('t (s)','Interpreter','none');
title(sprintf('%s [%s]',quantityLabel,unitLabel),'Interpreter','none');
set(gca,'FontName','Times New Roman','FontSize',12,'LineWidth',1.0);

save_figure_pair(fig, outDir, baseName);
close(fig);

[R,T] = meshgrid(r,t);
data = [R(:), T(:), Z(:)];
write_matrix_compat(data, fullfile(outDir,[baseName,'.txt']));
end

%% ========================================================================
function save_time_curve(outDir, baseName, t, y, quantityLabel, unitLabel)
% Save one scalar-vs-time curve and a three-column t,0,value TXT file.
fig = figure('Visible','off','Color','w','Position',[100 100 900 620]);
plot(t,y,'LineWidth',1.5);
grid on;
box on;
xlabel('t (s)','Interpreter','none');
ylabel(sprintf('%s [%s]',quantityLabel,unitLabel),'Interpreter','none');
title(quantityLabel,'Interpreter','none');
set(gca,'FontName','Times New Roman','FontSize',12,'LineWidth',1.0);

save_figure_pair(fig, outDir, baseName);
close(fig);

data = [t(:), zeros(numel(t),1), y(:)];
write_matrix_compat(data, fullfile(outDir,[baseName,'.txt']));
end

%% ========================================================================
function save_phase_portrait(outDir, baseName, u, v, t)
fig = figure('Visible','off','Color','w','Position',[100 100 760 650]);
plot(u,v,'LineWidth',1.1);
grid on;
box on;
xlabel('u_{outer} (m)');
ylabel('v_{outer} (m/s)');
title('Outer-surface phase portrait');
set(gca,'FontName','Times New Roman','FontSize',12,'LineWidth',1.0);
save_figure_pair(fig, outDir, baseName);
close(fig);

% columns: x=u_outer, y=v_outer, value=time
data = [u(:), v(:), t(:)];
write_matrix_compat(data, fullfile(outDir,[baseName,'.txt']));
end

%% ========================================================================
function save_fft_spectrum(outDir, baseName, t, u)
% FFT on uniformly requested output grid after transient removal.
dt = mean(diff(t));
y = u(:)-mean(u(:));
N = numel(y);
Y = fft(y);
A2 = abs(Y/N);
Nhalf = floor(N/2)+1;
A1 = A2(1:Nhalf);
if Nhalf > 2
    A1(2:end-1) = 2*A1(2:end-1);
end
f = (0:Nhalf-1).'/(N*dt);

fig = figure('Visible','off','Color','w','Position',[100 100 900 620]);
plot(f,A1,'LineWidth',1.2);
grid on;
box on;
xlabel('f (Hz)');
ylabel('Amplitude (m)');
title('Outer-surface displacement FFT');
set(gca,'FontName','Times New Roman','FontSize',12,'LineWidth',1.0);
save_figure_pair(fig, outDir, baseName);
close(fig);

% columns: frequency, 0, amplitude
data = [f, zeros(numel(f),1), A1];
write_matrix_compat(data, fullfile(outDir,[baseName,'.txt']));
end

%% ========================================================================
function save_poincare(outDir, baseName, t, u, v, Tforce, tStart)
% Interpolate the solution at integer forcing periods after transients.
n0 = ceil(tStart/Tforce);
n1 = floor(t(end)/Tforce);
if n1 <= n0
    return;
end
tp = ((n0:n1).'*Tforce);
up = interp1(t,u,tp,'pchip');
vp = interp1(t,v,tp,'pchip');

fig = figure('Visible','off','Color','w','Position',[100 100 760 650]);
plot(up,vp,'o','MarkerSize',4.5,'LineWidth',1.0);
grid on;
box on;
xlabel('u_{outer} (m)');
ylabel('v_{outer} (m/s)');
title('Stroboscopic Poincare section');
set(gca,'FontName','Times New Roman','FontSize',12,'LineWidth',1.0);
save_figure_pair(fig, outDir, baseName);
close(fig);

% columns: x=u_outer, y=v_outer, value=sample time
data = [up(:), vp(:), tp(:)];
write_matrix_compat(data, fullfile(outDir,[baseName,'.txt']));
end

%% ========================================================================
function save_figure_pair(fig, outDir, baseName)
pngFile = fullfile(outDir,[baseName,'.png']);
figFile = fullfile(outDir,[baseName,'.fig']);
try
    exportgraphics(fig,pngFile,'Resolution',300);
catch
    print(fig,pngFile,'-dpng','-r300');
end
savefig(fig,figFile);
end

%% ========================================================================
function write_matrix_compat(data, fileName)
% Numeric-only tab-delimited output. Uses writematrix when available and
% falls back to dlmwrite for older MATLAB releases.
if exist('writematrix','file') == 2
    writematrix(data, fileName, 'Delimiter','tab');
else
    dlmwrite(fileName, data, 'delimiter','\t', 'precision','%.16e'); %#ok<DLMWT>
end
end

%% ========================================================================
function write_parameter_summary(fileName, p, Tforce, tEnd, NtOut, minLR, minLT, minJ)
fid = fopen(fileName,'w');
if fid < 0
    warning('Could not create parameter summary file.');
    return;
end
cleanupObj = onCleanup(@() fclose(fid)); %#ok<NASGU>

fprintf(fid,'Spherical finite-deformation visco-hyperelastic strain-gradient cavity model\n');
fprintf(fid,'All quantities use SI units.\n\n');
fprintf(fid,'a = %.16e m\n',p.a);
fprintf(fid,'b = %.16e m\n',p.b);
fprintf(fid,'rho0 = %.16e kg/m^3\n',p.rho0);
fprintf(fid,'mu = %.16e Pa\n',p.mu);
fprintf(fid,'kappa = %.16e Pa\n',p.kappa);
fprintf(fid,'ell_g = %.16e m\n',p.ell_g);
fprintf(fid,'g = %.16e Pa m^2\n',p.g);
fprintf(fid,'eta = %.16e Pa s\n',p.eta);
fprintf(fid,'zeta = %.16e Pa s\n',p.zeta);
fprintf(fid,'gamma_g = %.16e\n',p.gamma_g);
fprintf(fid,'p0 = %.16e Pa\n',p.p0);
fprintf(fid,'DeltaP = %.16e Pa\n',p.DeltaP);
fprintf(fid,'Omega = %.16e rad/s\n',p.Omega);
fprintf(fid,'forcing_period = %.16e s\n',Tforce);
fprintf(fid,'nRamp = %.16e periods\n',p.nRamp);
fprintf(fid,'nPeriod = %.16e periods\n',p.nPeriod);
fprintf(fid,'tEnd = %.16e s\n',tEnd);
fprintf(fid,'N_radial = %d\n',p.N);
fprintf(fid,'N_time_output = %d\n',NtOut);
fprintf(fid,'dr = %.16e m\n',p.dr);
fprintf(fid,'minimum accepted lambda_r = %.16e\n',minLR);
fprintf(fid,'minimum accepted lambda_theta = %.16e\n',minLT);
fprintf(fid,'minimum accepted J = %.16e\n',minJ);
end
