clc; clear; close all;

%% parameters
N = 50;     % number of realizations
M = 20;     % number of PA per WG

delta_x = 0.5;  % m, spacing between PA
Dx = delta_x*M; % m, side of area
Dy = 10;        % m
h = 3;          % m, height of PA

fc = 30e9;      % [Hz]
c = 3e8;        % speed of light, [m/s]
eta = c / (4*pi*fc); % free space path loss
n_eff = 1.4;         % effective refractive index
lambda = c/fc;
lambda_g = lambda / n_eff;
alpha = 0.18;   % waveguide attenuation coeff


%% powers
Pn = 50;   % dBm, noise power
P = 0:5:40; % dBm, signal power

Pn_lin = 10^((Pn-30)/10); % W
P_lin = 10.^((P-30)/10); % W

%% user position
phi_u = [Dx*rand(N, 1), Dx*rand(N, 1) - Dx/2, zeros(N, 1)]; % user positions

%% PA coordinates
p_0 = [0,0,0];  % origin
PA_positions = linspace(Dx/M, Dx, M); % position of the pinching antennas along the waveguide
PA_positions = [PA_positions.', zeros(M,1), h*ones(M,1)];

%% compute channel gains and rate
rates_mean = zeros(length(P),1);

for i = 1:length(P)    % all the powers
    rate = zeros(1,N);
    for n = 1:N        % all the realizations
        ch_gains = zeros(M, 1); % channel vector of the user/realization for all PA positions
        for m = 1:M    % all the PA positions
            ch_gains(m) = eta * exp(-1j*(2*pi/lambda)*norm(phi_u(n)-PA_positions(m))) / norm(phi_u(n)-PA_positions(m)) * exp((-alpha-1j*2*pi/lambda_g)*norm(p_0-PA_positions(m)));
        end
        optimal_PA = max(abs(ch_gains)); % get the optimal PA
        gamma = optimal_PA * P_lin(i) / Pn_lin;
        rate(n) = log2(1 + gamma);
    end
    rates_mean(i) = mean(rate);
end

%% plot user rate vs power
figure;
plot(P, rates_mean, 'Marker','o');
xlabel('Signal Power [dBm]'); ylabel('Rate');
title('Rate vs Power');
grid on;

%% plot users and PA positions
% figure;
% scatter(phi_u(:,1), phi_u(:,2), "DisplayName", "Users"); hold on;
% scatter(PA_positions(:,1), PA_positions(:,2), 'magenta', "DisplayName", "PA positions");
% title('Users and pinching antenna positions');
% grid on;
% legend;