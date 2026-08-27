clear; clc; close all;

%% parameters
M = 20;     % number of PA per WG
K = 3;      % number of WG
U = 2;      % number of users

delta_x = 0.5;  % [m], spacing between PA
delta_y = 3;    % [m], spacing between waveguides
Dx = delta_x*M; % [m], side of area
Dy = delta_y*K; % [m]
h = 3;          % [m], height of PA

%% PA coordinates
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

%% realizations
phi_u = [Dx*rand(U, 1), Dy*rand(U, 1) - Dy/2, zeros(U, 1)]; % user positions

%% plot positions

figure;
scatter3(PA_pts(:,1), PA_pts(:,2), PA_pts(:,3), "ro", "filled", "DisplayName", "PA posiitons"); 
hold on;
scatter3(phi_u(:,1),phi_u(:,2),phi_u(:,3), "bo", "filled", "DisplayName", "Users");
title("PA positions with " + U + " ramdom users");
xlabel("x (m)"); ylabel("y (m)"); zlabel("z (m)");
xlim([0,Dx]); ylim([-Dy/2, Dy/2]); zlim([0,h+1]);
yticks(y);
view(45, 45);
legend;