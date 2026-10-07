clc; clear; close all;

%% parameters
N = 100;    % number of realizations
M = 20;     % number of PA per WG
K = 3;      % number of WG
U = 2;      % number of users

delta_x = 0.5;  % m, spacing between PA
delta_y = 5;    % m, spacing between waveguides
Dx = delta_x*M; % m, side of area
Dy = delta_y*K; % m
h = 3;          % m, height of PA

fc = 30e9;      % Hz
c = 3e8;        % speed of light, m/s
eta = c/(4*pi*fc);  % free space path loss
n_eff = 1.4;        % effective refractive index
lambda = c/fc;
lambda_g = lambda / n_eff;
alpha = 0.18;   % waveguide attenuation coeff

%% powers
Pn = -50;   % dBm, noise power
P = 0:5:40; % dBm, signal power

Pn_lin = 10^((Pn-30)/10); % W
P_lin = 10.^((P-30)/10); % W

Pu = P_lin / U;

%% PA coordinates
p_0 = [0,0,0];  % origin

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

PA_pts = reshape(PA_coords, [], 3); % KM x 3

%% users positions
users_pos = [Dx*rand(N*U, 1), Dy*rand(N*U, 1) - Dy/2, zeros(N*U, 1)]; % user positions

%% compute channel gains
C = zeros(K,M,U,N);

for n = 1:N
    for u = 1:U 
        idx = (n-1)*U + u;
        for k = 1:K     % all the WG
            for m = 1:M     % all the PA positions
                PA_pos = reshape(PA_coords(k,m,:),1,3); % PA coord as a vector
                d = norm(users_pos(idx,:) - PA_pos);  % distance between user and PA
                C(k,m,u,n) = eta * exp(-1j*(2*pi/lambda)*d) / d * exp((-alpha-1j*2*pi/lambda_g)*norm(p_0 - PA_pos));
            end 
        end
    end
end

%% baseline
% build S from max channel gain for each waveguide
fprintf("Baseline \n");
rate_baseline_mean = zeros(length(P),1);

for p = 1:length(P)
    rate_baseline = zeros(N,1);

    for n = 1:N
        Cn = C(:,:,:,n);

        % get the indexes of the max channel gain for every waveguide
        indexes_S = ones(K,1);
        for k = 1:K
            [~, i] = max(abs(C(k,:,:,n)),[],2);
            indexes_S(k) = max(i);
        end
               
        H_ZF = zeros(K,U); % build H from C and S
        for k = 1:K
            for u = 1:U
                H_ZF(k,u) = C(k,indexes_S(k),u,n);
            end
        end

        F_ZF = H_ZF / (H_ZF' * H_ZF);

        user_rate = zeros(U,1);
        for u = 1:U
            gain = abs(H_ZF(:,u)' * F_ZF(:,u))^2 / norm(F_ZF(:,u))^2;
            user_rate(u) = log2(1 + (P_lin(p)/U) / Pn_lin * gain);
        end

        rate_baseline(n) = sum(user_rate);
    end

    rate_baseline_mean(p) = mean(rate_baseline);

    fprintf("  P=%d dBm,  R_baseline=%.3f \n", P(p), rate_baseline_mean(p));
end


%% ZF precoding, optimization over S
fprintf("Optimization over S \n");
rate_fullsearch_mean = zeros(length(P),1);
rate_fmincon_mean = zeros(length(P),1);

for p = 1:length(P)
    rate_fullsearch = zeros(N,1);
    rate_fmincon = zeros(N,1);

    for n = 1:N
        Cn = C(:,:,:,n);

        [best_sum_rate_fullsearch, best_S_fullsearch, best_indexes_fullsearch] = ...
            position_optim_ZF_fullsearch(Cn,K,M,U,P_lin(p),Pn_lin);
        [best_sum_rate_fmincon, best_S_fmincon, best_indexes_fmincon] = ...
            position_optim_ZF_fmincon(Cn,K,M,U,P_lin(p),Pn_lin);

        rate_fullsearch(n) = best_sum_rate_fullsearch;
        rate_fmincon(n) = best_sum_rate_fmincon;

    end
    rate_fullsearch_mean(p) = mean(rate_fullsearch);
    rate_fmincon_mean(p) = mean(rate_fmincon);

    fprintf("  P=%d dBm,  R_fulls=%.3f, R_fmincon=%.3f \n", P(p), ...
        rate_fullsearch_mean(p), rate_fmincon_mean(p));
    fprintf("    Best indexes: ");
    fprintf("[ "); fprintf("%g ", best_indexes_fullsearch(1:end)); fprintf("] ");
    fprintf("[ "); fprintf("%g ", best_indexes_fmincon(1:end)); fprintf("] ");
    fprintf("\n")

end

%% plot sum rates
figure;
plot(P, rate_baseline_mean, "-r*", "LineWidth", 1.5, "DisplayName", "baseline"); hold on;
plot(P, rate_fullsearch_mean, "-go", "LineWidth", 1.5, "DisplayName", "full search"); hold on;
plot(P, rate_fmincon_mean, "-mo", "LineWidth", 1.5, "DisplayName", "fmincon");
xlabel("Transmit Power [dBm]"); ylabel("Sum Rate");
legend("Location", "northwest"); grid on;
title("Sum Rate vs POwer");

