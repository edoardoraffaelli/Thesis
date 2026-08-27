clc; clear; close all;

%% parameters
M = 20;     % number of PA per WG
K = 3;      % number of WG
N = 100;    % number of realizations
U = 2;      % number of users

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

Pu = P_lin / U;

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
phi_u = [Dx*rand(N*U, 1), Dy*rand(N*U, 1) - Dy/2, zeros(N*U, 1)];     % user positions

%% max, 1 PA for both users
rates_max_mean = zeros(length(P),1);

for p = 1:length(P)    % all the powers
    rate_max = zeros(N,1);
    
    for n = 1:N
        idx = (n-1)*U + (1:U);      
        users_pos = phi_u(idx,:);   % U users at the time

        ch_gain_max = 0;
        gamma_max = 0;
        user_rate_max = zeros(U,1);

        for u = 1:U        % all the users
    
            ch_gains = zeros(K,M); % channel vector of the user for all PA positions
            h_optimal = zeros(K,1);
    
            for k = 1:K    % all the WG
                for m = 1:M     % all the PA positions
                    PA_pos = reshape(PA_coords(k,m,:),1,3); % PA coord as a vector
                    d = norm(users_pos(u) - PA_pos);  % distance between user and PA
                    ch_gains(k,m) = eta * exp(-1j*(2*pi/lambda)*d) / d * exp((-alpha-1j*2*pi/lambda_g)*norm(p_0 - PA_pos));
                end 
                [~, PA_index] = max(abs(ch_gains(k,:))); % K best PA, one for each WG 
                h_optimal(k) = ch_gains(k, PA_index); 
            end
    
            ch_gain = max(abs(h_optimal));
            gamma_max = ch_gain^2 * P_lin(p) / Pn_lin;
            user_rate_max = log2(1 + gamma_max);
        end

    end
    
    rates_max_mean(p) = mean(user_rate_max);
end

%% MRT, K PA
rate_MRT_mean = zeros(length(P),1);

for p = 1:length(P)    % all the powers
    
    rate_MRT = zeros(N,1);
    
    for n = 1:N
        idx = (n-1)*U + (1:U);      
        users_pos = phi_u(idx,:);

        ch_gain_MRT = 0;
        gamma_MRT = 0;
        user_rate_MRT = zeros(U,1);
    
        for u = 1:U        % all the realizations
    
            ch_gain_MRT = zeros(K,M); % channel vector of the user for all PA positions
            ch_gain_optimal_MRT = zeros(K,1);
    
            for k = 1:K    % all the WG
                for m = 1:M     % all the PA positions
                    PA_pos = reshape(PA_coords(k,m,:),1,3); % PA coord as a vector
                    d = norm(users_pos(u) - PA_pos);  % distance between user and PA
                    ch_gain_MRT(k,m) = eta * exp(-1j*(2*pi/lambda)*d) / d * exp((-alpha-1j*2*pi/lambda_g)*norm(p_0 - PA_pos));
                end 
                [~, PA_index] = max(abs(ch_gain_MRT(k,:))); % K best PA, one for each WG 

                % for each waveguide, take the PA with maximum ch_gain
                ch_gain_optimal_MRT(k) = ch_gain_MRT(k, PA_index); 
            end
    
            ch_gain_MRT = norm(ch_gain_optimal_MRT);
            gamma_MRT = ch_gain_MRT^2 * P_lin(p) / Pn_lin;
            user_rate_MRT(u) = log2(1 + gamma_MRT);
        end

        rate_MRT(n) = mean(user_rate_MRT);
    end
    
    rate_MRT_mean(p) = mean(rate_MRT);
end


%% ZF, exhaustive search
sum_rate_ZF_mean = zeros(length(P), 1);

for p = 1:length(P)
    sum_rate_ZF = zeros(N, 1);

    for n = 1:N
        idx = (n-1)*U + (1:U);      
        users_pos = phi_u(idx,:);

        ch_gains = zeros(K,M,U);
        
        for u = 1:U
            for k = 1:K    % all the WG
                for m = 1:M     % all the PA positions
                    PA_pos = reshape(PA_coords(k,m,:),1,3); % PA coord as a vector
                    d = norm(users_pos(u) - PA_pos);  % distance between user and PA
                    ch_gains(k,m,u) = eta * exp(-1j*(2*pi/lambda)*d) / d * exp((-alpha-1j*2*pi/lambda_g)*norm(p_0 - PA_pos));
                end  
            end
        end
    
        % exhaustive search over all (m1, m2, m3) 
        best_sum_rate = 0;
        best_m = ones(1, K);   % best PA index per waveguide
    
        for m1 = 1:M
            for m2 = 1:M
                for m3 = 1:M
                    m_sel = [m1, m2, m3];   % one PA index per waveguide k
    
                    H_ZF = zeros(K, U); % row k uses PA m_sel(k) for ALL users
                    for u = 1:U
                        for k = 1:K
                            H_ZF(k, u) = ch_gains(k, m_sel(k), u);
                        end
                    end
                    
                    F_ZF = H_ZF / (H_ZF' * H_ZF);
                    % F_ZF = H_ZF * inv(H_ZF' * H_ZF);
    
                    % compute user rate and sum rate
                    user_rate_ZF = zeros(U,1);
                    for u = 1:U
                        h_gain = abs(H_ZF(:,u)' * F_ZF(:,u))^2 / norm(F_ZF(:,u))^2;
                        user_rate_ZF(u) = log2(1 + Pu(p) / Pn_lin * h_gain);
                    end
                    sum_rate_tmp = sum(user_rate_ZF(:));
                    
                    % check if is best rate
                    if sum_rate_tmp > best_sum_rate  
                        best_sum_rate = sum_rate_tmp;
                        best_m = m_sel;
                        % fprintf('Optimal PA indices per waveguide: m = [%d, %d, %d]\n', m1,m2,m3);
                    end
                end
            end
        end
        
        sum_rate_ZF(n) = best_sum_rate;
    end

    sum_rate_ZF_mean(p) = mean(sum_rate_ZF);
end


%% plot
figure;
plot(P, rates_max_mean, "-ro", "LineWidth", 1.5, "DisplayName", "max"); hold on;
plot(P, rate_MRT_mean, "-bo", "LineWidth", 1.5, "DisplayName", "MRT");
plot(P, sum_rate_ZF_mean, "-go", "LineWidth", 1.5, "DisplayName", "ZF");
xlabel("Signal Power [dBm]"); ylabel("Rate");
title("Rate vs Power");
legend("Location", "northwest");
grid on;


