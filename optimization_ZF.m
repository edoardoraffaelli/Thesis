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

fc = 30e9;      % [Hz]
c = 3e8;        % speed of light [m/s]
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
phi_u = [Dx*rand(N*U, 1), Dy*rand(N*U, 1) - Dy/2, zeros(N*U, 1)]; % user positions

%% compute channel gains
ch_gains = zeros(K, M, U);

for u = 1:U*N
    for k = 1:K
        for m = 1:M
            PA_pos = reshape(PA_coords(k,m,:), 1, 3);
            d = norm(phi_u(u,:) - PA_pos);
            ch_gains(k,m,u) = eta * exp(-1j*(2*pi/lambda)*d) / d ...
                * exp((-alpha-1j*2*pi/lambda_g)*norm(p_0-PA_pos));
        end
    end
end

%% ZF, exhaustive search
fprintf("Full search: \n");

sum_rate_ZF_full_search_mean = zeros(length(P),1);
all_rates = zeros(M, M, M);

for p = 1:length(P)
    
    for n = 1:N

        idx = (n-1)*U + (1:U) ;
        sum_rate_ZF_full_search = zeros(N,1);

        % exhaustive search over all (m1, m2, m3) 
        best_sum_rate = 0;
        best_m = ones(1, K);   % best PA index per waveguide
    
        for m1 = 1:M
            for m2 = 1:M
                for m3 = 1:M
                    m_sel = [m1, m2, m3];   % one PA index per waveguide k

                    H_ZF = zeros(K, U);
                    for u = 1:U
                        for k = 1:K
                            H_ZF(k, u) = ch_gains(k, m_sel(k), idx(u));
                        end
                    end

                    F_ZF = H_ZF / (H_ZF' * H_ZF);

                    % compute user rate and sum rate
                    user_rate_ZF = zeros(U,1);
                    for u = 1:U
                        h_gain = abs(H_ZF(:,u)' * F_ZF(:,u))^2 / norm(F_ZF(:,u))^2;
                        user_rate_ZF(u) = log2(1 + Pu(p) / Pn_lin * h_gain);
                    end
                    sum_rate_tmp = sum(user_rate_ZF(:));

                    if p == (length(P)-1)
                        all_rates(m1,m2,m3) = sum_rate_tmp;
                    end

                    % check if is best rate
                    if sum_rate_tmp > best_sum_rate  
                        best_sum_rate = sum_rate_tmp;
                        best_m = m_sel;
                        % fprintf('Optimal PA indices per waveguide: m = [%d, %d, %d]\n', m1,m2,m3);
                    end
                end
            end
        end
    
        sum_rate_ZF_full_search(n) = best_sum_rate;
        % fprintf("Power=%d dBm, optimal PA indices: m = [%d %d %d]\n", P(p), best_m(1), best_m(2), best_m(3));
    
    end
    
    % print the used PA positions for the last realization
    fprintf("Power=%d dBm, optimal PA indices: m = [%d %d %d]\n", P(p), best_m(1), best_m(2), best_m(3));

    sum_rate_ZF_full_search_mean = mean(sum_rate_ZF_full_search);
end

%% function to compute sum rate

% negative sum rate as function of vector s, K*M x 1
function neg_sum_rate = neg_sum_rate_S(S, ch_gains, K, M, U, idx, Pu, Pn_lin)
    % build H_ZF form S
    H_ZF = zeros(K, U);
    for k = 1:K
        for u = 1:U
            H_ZF(k,u) = S(k,:) * ch_gains(k,:,idx(u)).';
        end
    end
    
    F_ZF = H_ZF / (H_ZF' * H_ZF);
    
    % negative sum rate
    user_rate = zeros(U,1);
    for u = 1:U
        h_gain = abs(H_ZF(:,u)' * F_ZF(:,u))^2 / norm(F_ZF(:,u))^2;
        user_rate(u) = log2(1 + Pu / Pn_lin * h_gain);
    end
    neg_sum_rate = -sum(user_rate);
end

%% constraints on S

% function for non linear constraints
function [ineqnonlin, eqnonlin] = S_constraints(S)
    % Nonlinear constraints, specified as a function handle or function name. 
    % nonlcon is a function that accepts a vector or array x and returns two arrays, 
    % ineqnonlin(x) and eqnonlin(x).

    ineqnonlin = [];    
    eqnonlin = [];   % Compute nonlinear equalities at x, no inequality constraints in this case
    % eqnonlin = sum(S - S.^2, "all");
end


% equality constraint -> Aeq * s = beq
Aeq = zeros(K, K*M);
for k = 1:K
    % indices in the unrolled vector corresponding to row k of S
    Aeq(k, k:K:end) = 1;    % every K-th element starting from k
end
beq = ones(K, 1); % each row sums to 1


% initial points for S, non-binary random matrix
S_0 = rand(K, M);
S_0 = S_0 ./ sum(S_0,2);  % normalization of each row


% lower upper bounds
lb = zeros(size(S_0)); 
up = ones(size(S_0));

% optimization options
% options = optimoptions("fmincon", "Display", "off");
options = optimoptions('fmincon', ...
    'Algorithm', 'interior-point', ...
    'MaxIterations', 1e6, ...
    'MaxFunctionEvaluations', 1e6, ...
    'StepTolerance', 1e-12, ...
    'OptimalityTol', 1e-9, ...
    'Display', 'none');

%% ZF, optimization with fmincon
fprintf("Fmincon: \n");

sum_rate_ZF_fmincon_mean = zeros(length(P), 1);

for p = 1:length(P)

    for n = 1:N

        idx = (n-1)*U + (1:U);

        sum_rate_ZF_fmincon = zeros(N,1);

        % wrap rate and constraints
        rate = @(S) neg_sum_rate_S(S, ch_gains, K, M, U, idx, Pu(p), Pn_lin);
        nonlcon = @(S) S_constraints(S); 
    
        S_opt = fmincon(rate, S_0, [], [], Aeq, beq, lb, up, nonlcon, options);
    
        [~, s_opt_index] = max(S_opt, [], 2); % from matrix S to indexes of the maxima
    
        S_final = zeros(K,M);
        for k = 1:K
            S_final(k,s_opt_index(k)) = 1;
        end
    
        sum_rate_ZF_fmincon(n) = -neg_sum_rate_S(S_final, ch_gains, K, M, U, idx, Pu(p), Pn_lin);
    
    end 

    % print the used PA positions for the last realization
    fprintf("Power=%d dBm, optimal PA indices: m = [%d %d %d]\n", P(p), s_opt_index(1), s_opt_index(2), s_opt_index(3));

    sum_rate_ZF_fmincon_mean = mean(sum_rate_ZF_fmincon);
end


%% plot sum rates
figure;
plot(P, sum_rate_ZF_full_search_mean, "-go", "LineWidth", 1.5, "DisplayName", "Full Search"); hold on;
plot(P, sum_rate_ZF_fmincon, "-ms", "LineWidth", 1.5, "DisplayName", "fmincon");
xlabel("Transmit Power (dBm)"); ylabel("Sum Rate");
legend("Location", "northwest"); grid on;
title("Sum Rate with ZF");

