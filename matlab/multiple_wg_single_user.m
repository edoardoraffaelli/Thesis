clc; clear; close all;

%% parameters
M = 20;     % number of PA per WG
K = 3;      % number of WG
N = 100;    % number of realizations

delta_x = 0.5;  % [m], spacing between PA
delta_y = 5;    % [m]
Dx = delta_x*M; % [m], side of area
Dy = delta_y*K; % [m]
h = 3;          % [m], height of PA

fc = 30e9;      % [Hz]
c = 3e8;        % speed of light, [m/s]
eta = c/(4*pi*fc);  % free space path loss
n_eff = 1.4;        % effective refractive index
lambda = c/fc;
lambda_g = lambda / n_eff;
alpha = 0.18;   % waveguide attenuation coeff

%% powers
Pn = -50;  % [dBm], noise power
P = 0:5:40; % [dBm], signal power

Pn_lin = 10^((Pn-30)/10);   % [W]
P_lin = 10.^((P-30)/10);

%% PA coordinates
p_0 = [0,0,0];

x = (0:M-1) * delta_x; 
PA_x = ones(K,1)*x; % K x M

half_y = (K-1)/2;
y = ((0:K-1) - half_y)*delta_y;
PA_y = y'*ones(1,M); % K x M

PA_z = h*ones(K,M); % K x M

PA_coords = zeros(K, M, 3);
PA_coords(:, :, 1) = PA_x;  
PA_coords(:, :, 2) = PA_y;  
PA_coords(:, :, 3) = PA_z;  

PA_pts = reshape(PA_coords, [], 3);      % KM x 3

%% user positions
phi_u = [Dx*rand(N, 1), Dy*rand(N, 1) - Dy/2, zeros(N, 1)];

%% compute channel gains and rates
rates_mean = zeros(length(P),1);

for p = 1:length(P)    % all the powers
    rate = zeros(N,1);
    gamma = 0;
    ch_gain_max = zeros(N,1);

    PA_index = zeros(N,K);
    PA_max_index = zeros(N,1);

    for n = 1:N % all the realizations

        ch_gains = zeros(K,M); % channel vector of the user for all PA positions
        h_optimal = zeros(K,1);

        for k = 1:K    % all the WG
            for m = 1:M     % all the PA positions
                PA_pos = reshape(PA_coords(k,m,:),1,3);  % PA coord as a vector
                d = norm(phi_u(n,:) - PA_pos);  % distance between user and PA
                ch_gains(k,m) = eta * exp(-1j*(2*pi/lambda)*d) / d * exp((-alpha-1j*2*pi/lambda_g)*norm(p_0 - PA_pos));
            end

            % take the best PA from each waveguide and save the indexes
            [h_optimal(k), PA_index(n,k)] = max(abs(ch_gains(k,:))); % K best PA, one for each WG 
        end
        
        % take the best PA between all waveguides and save the index of the waveguide
        [ch_gain_max(n), PA_max_index(n)] = max(h_optimal);
        gamma = ch_gain_max(n)^2 * P_lin(p) / Pn_lin;
        rate(n) = log2(1 + gamma);

    end

    rates_mean(p) = mean(rate);
end


%% plot sum rate vs power
figure;
plot(P, rates_mean, "-o"); hold on;
xlabel("Signal Power [dBm]"); ylabel("Rate");
title("Rate vs Power");
grid on;

%% plot positions of PA and the best PA

% PA_best_pts = zeros(K*N,3); % the 3 best PA, one for each wg, for each realization
% for n = 1:N
%     for k = 1:K
%         PA_best_pts((n-1)*K+k,:) = PA_coords(k, PA_index(k,n), :);
%     end
% end
% 
% PA_max_pt = zeros(N,3); % one max PA for each realization
% for n = 1:N
%     k = PA_max_index(n);
%     PA_max_pt(n,:) = PA_coords(k, PA_index(k,n), :);
% end
% 
% figure;
% scatter3(phi_u(:,1),phi_u(:,2),phi_u(:,3), "bo", "filled"); hold on;
% scatter3(PA_pts(:,1), PA_pts(:,2), PA_pts(:,3), "go", "filled");
% scatter3(PA_best_pts(:,1), PA_best_pts(:,2), PA_best_pts(:,3), "r*");
% scatter3(PA_max_pt(:,1), PA_max_pt(:,2), PA_max_pt(:,3), "b*");
% title("PA positions");
% xlabel("x [m]"); ylabel("y [m]"); zlabel("z [m]");
% xlim([0, Dx]); ylim([-Dy/2, Dy/2]); zlim([0, h+1]);
% yticks(y);
% grid on;
% view(45, 45);

