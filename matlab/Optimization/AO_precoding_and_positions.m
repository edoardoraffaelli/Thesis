clc; clear; close all;

%% parameters
N = 100;     % number of channel realizations to average over
M = 20;     % number of PA per WG
K = 3;      % number of WG
U = 2;      % number of users

delta_x = 0.5;  % m, spacing between PA
delta_y = 3.0;  % m, spacing between waveguides
Dx = delta_x*M; % m, side of area
Dy = delta_y*K; % m
h = 3;          % m, height of PA

fc = 30e9;      % Hz
c = 3e8;        % speed of light
eta = c/(4*pi*fc);  % free space path loss
n_eff = 1.4;        % effective refractive index
lambda = c/fc;
lambda_g = lambda / n_eff;
alpha = 0.18;   % waveguide attenuation coeff

%% powers
Pn = -50; % dBm, noise power
P = 0:5:40; % dBm, signal power

Pn_lin = 10^((Pn-30)/10); % W
P_lin = 10.^((P-30)/10); % W

SNR = P_lin / Pn_lin;

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

%% users positions
users_pos = [Dx*rand(N*U, 1), Dy*rand(N*U, 1) - Dy/2, zeros(N*U, 1)]; % user positions

%% compute all channel gains
C = zeros(K,M,U,N);
for n = 1:N
    for u = 1:U
        idx = (n-1)*U + u;
        for k = 1:K
            for m = 1:M
                PA_pos = reshape(PA_coords(k,m,:),1,3);
                d = norm(users_pos(idx,:) - PA_pos);
                C(k,m,u,n) = eta * exp(-1j*(2*pi/lambda)*d) / d * exp((-alpha-1j*2*pi/lambda_g)*norm(p_0 - PA_pos));
            end
        end
    end
end

%% AO settings
ao_max_iter = 20;
ao_tol = 1e-6;

%% initial activation for the activations S_0
rand_activations = randi(M, K, 1);
S_0 = zeros(K, M);
for k = 1:K
    S_0(k, rand_activations(k)) = 1;
end

%% baseline: no optimization 
% build W=H and S from max channel gain for each waveguide
fprintf("Baseline: no optimization \n");

rate_baseline_mean = zeros(length(P),1);

for p = 1:length(P)
    rate_baseline = zeros(N,1);

    for n = 1:N     
        % initialization: get the indexes of the max channel gain for every waveguide
        indexes_S = ones(K,1);
        for k = 1:K
            [~, i] = max(abs(C(k,:,:,n)),[],2);
            indexes_S(k) = max(i);
        end

        H = zeros(K,U); % build H from C and S
        for k = 1:K
            for u = 1:U
                H(k,u) = C(k,indexes_S(k),u,n);
            end
        end

        W_0 = H;
        W_0 = W_0 ./ norm(W_0, "fro") * sqrt(P_lin(p));

        [sum_rate, user_rates] = compute_sum_rate(H,W_0,U,Pn_lin);
        rate_baseline(n) = sum_rate;
    end

    rate_baseline_mean(p) = mean(rate_baseline);

    fprintf("P=%d dBm \n  R_baseline_0 = %.3f\n", P(p), rate_baseline_mean(p));

    P_tx = 10*log10(norm(W_0, "fro")^2) + 30;
    fprintf("  Tx power = %.1f dBm\n", P_tx);
    fprintf("  last W = [ "); fprintf("%.3f ", abs(W_0)); fprintf("] \n");
    fprintf("  last sum rate: %.3f \n", rate_baseline(end));
    fprintf("  last user_rataes: ")
    fprintf(" [ "); fprintf("%.3f ", user_rates); fprintf("] \n");
end


%% baseline: optimize W 
% build S from max channel gain for each waveguide and optimize W

fprintf("\n Baseline: optimize W \n");

rate_baseline_W_mean = zeros(length(P),1);

for p = 1:length(P)
    rate_baseline = zeros(N,1);

    for n = 1:N
        % initialization, get the indexes of the max channel gain for every waveguide
        indexes_S = ones(K,1);
        for k = 1:K
            [~, i] = max(abs(C(k,:,:,n)),[],2);
            indexes_S(k) = max(i);
        end

        H = zeros(K,U); % build H from C and S
        for k = 1:K
            for u = 1:U
                H(k,u) = C(k,indexes_S(k),u,n);
            end
        end
       
        % optimize the precoding weights with WMMSE
        [W_final, R_hist, user_rates_final] = precoding_optim_WMMSE(H, P_lin(p), Pn_lin, 30, 1e-6);
        rate_baseline(n) = R_hist(end);
    end

    rate_baseline_W_mean(p) = mean(rate_baseline);

    fprintf("P=%d dBm \n  R_baseline = %.3f\n", P(p), rate_baseline_mean(p));

    P_tx = 10*log10(norm(W_final, "fro")^2) + 30;
    fprintf("  Tx power = %.1f dBm\n", P_tx);
    fprintf("  last W = [ "); fprintf("%.3f ", abs(W_final)); fprintf("] \n");
    fprintf("  last sum rate: %.3f \n", R_hist(end));
    fprintf("  last user_rataes: ")
    fprintf(" [ "); fprintf("%.3f ", user_rates_final); fprintf("] \n");
end

%% baseline: optimize S 
% build W=H and optimize S

fprintf("\n Baseline: optimize S \n");

rate_baseline_S_mean = zeros(length(P),1);

for p = 1:length(P)
    rate_baseline = zeros(N,1);

    W_0 = H;
    W_0 = W_0 ./ norm(W_0, "fro") * sqrt(P_lin(p));

    for n = 1:N
        Cn = C(:,:,:,n);

        % get the indexes of the max channel gain for every waveguide
        indexes_S = ones(K,1);
        for k = 1:K
            [~, i] = max(abs(C(k,:,:,n)),[],2);
            indexes_S(k) = max(i);
        end

        S_0 = zeros(K, M);
        for k=1:K
            S_0(k, indexes_S(k)) = 1;
        end

        % optimize the positions
        [best_sum_rate, ~, ~] = position_optim_MMSE_fullsearch(Cn,S_0,W_0,K,M,U,Pn_lin);
        rate_baseline(n) = best_sum_rate;
    end

    rate_baseline_S_mean(p) = mean(rate_baseline);

    fprintf("P=%d dBm \n  R_baseline = %.3f\n", P(p), rate_baseline_mean(p));

    % P_tx = 10*log10(norm(W_0,"fro")^2) + 30;
    % fprintf("  Tx power = %.1f dBm\n",P_tx);
    fprintf("  last W = [ "); fprintf("%.3f ", abs(W_0)); fprintf("] \n");
    fprintf("  last sum rate: %.3f \n", R_hist(end));
    % fprintf("  last user_rataes: ")
    % fprintf(" [ "); fprintf("%.3f ", user_rates_final); fprintf("] \n");
end


%% AO
fprintf("\n AO \n");
rate_ao_mean = zeros(length(P),1);
num_iter_ao_mean = zeros(length(P),1);

for p = 1:length(P)
    rate_ao = zeros(N,1);
    num_iterations = zeros(N,1);

    for n = 1:N
        Cn = C(:,:,:,n);

        [S_ao, W_ao, sum_rate_ao, rate_hist_ao, user_rates_ao] = ...
            ao_optim_position_precoding(Cn,S_0,K,M,U,P_lin(p), Pn_lin, ao_max_iter, ao_tol);

        rate_ao(n) = sum_rate_ao;
        num_iterations(n) = numel(rate_hist_ao);
    end

    rate_ao_mean(p) = mean(rate_ao);
    num_iter_ao_mean(p) = mean(num_iterations);

    fprintf("P=%d dBm \n  R_AO = %.3f\n", P(p), rate_ao_mean(p));

    P_tx = 10*log10(norm(W_ao,"fro")^2) + 30;
    fprintf("  Tx power = %.1f dBm\n", P_tx); 
    fprintf("  last W = [ "); fprintf("%.3f ", abs(W_ao)); fprintf("] \n");
    fprintf("  last sum rate: %.3f \n", rate_ao(end));
    fprintf("  last user_rataes: ")
    fprintf("[ "); fprintf("%.3f ", user_rates_ao); fprintf("] \n");
    fprintf("  num iterations=%.1f \n", num_iter_ao_mean(p));
end

%% plot sum rates
figure;
plot(P, rate_baseline_mean, "-r*", "LineWidth", 1.5, "DisplayName", "No Optimization"); hold on;
plot(P, rate_baseline_S_mean, "-ms", "LineWidth", 1.5, "DisplayName", "Optimization over S"); 
plot(P, rate_baseline_W_mean, "-bs", "LineWidth", 1.5, "DisplayName", "Optimization over W"); 
plot(P, rate_ao_mean, "-go", "LineWidth", 1.5, "DisplayName", "AO over W and S");
xlabel("Transmit Power [dBm]"); ylabel("Sum Rate");
legend("Location", "northwest"); grid on;
title("Sum Rate vs Power");
% subtitle("with Alternating Optimization");

%% function
function [sum_rate, user_rates] = compute_sum_rate(H,W,U,Pn_lin)
    interf_plus_noise = sum(abs(H'*W).^2, 2) + Pn_lin; % sum_i |h_u^H w_i|^2 + Pn_lin
    e = ones(U,1) - (abs(diag(H'*W)).^2 ./ interf_plus_noise); % Ux1
    user_rates = -log2(e);
    sum_rate = sum(user_rates);
end

