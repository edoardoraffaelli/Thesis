function [S_opt, W_opt, rate_opt, rate_hist, user_rates_final] = ao_optim_position_precoding(C, S_0, K, M, U, P_lin, Pn_lin, max_ao_iter, tol)
% alternating optimization for joint PA activation (S) and MMSE beamforming (W)
%
%   [S_opt, W_opt, rate_opt, rate_hist, user_rates] = ...
%       ao_optim_position_precoding(C, S_init, K, M, U, P_lin, Pn_lin, max_ao_iter, tol)
%
%   C           : K x M x U effective channel tensor
%   S_0         : K x M, initial PA activation matrix (one 1 per row)
%   K, M, U     : number of waveguides, PA positions per WG, users
%   P_lin       : total tx power budget, sum_u ||w_u||^2 <= P_lin
%   Pn_lin      : noise power
%   max_ao_iter : max number of outer AO iterations (default 20)
%   tol         : convergence tolerance on sum-rate increase (default 1e-6)
%
%   S_opt       : K x M, final PA activation matrix
%   W_opt       : K x U, final precoder, RE-SOLVED for S_opt so that
%                 (S_opt, W_opt) are mutually consistent (see note below)
%   rate_opt    : sum rate achieved by (S_opt, W_opt)
%   rate_hist   : per-AO-iteration sum rate, i.e. rate(S_new, W) achieved
%                 by that iteration's S-step -- should be non-decreasing
%                 with 'fullsearch' (see convergence note below)
%   user_rates_final : individual user rate at the last iteration

    if nargin < 8 || isempty(max_ao_iter), max_ao_iter = 20; end
    if nargin < 9 || isempty(tol), tol = 1e-6; end

    S = S_0;
    rate_hist = zeros(max_ao_iter, 1);

    for ao_iter = 1:max_ao_iter

        % fix S, optimize over W 
        H = build_H_matrix(S, C, K, U);
        [W, ~, ~] = precoding_optim_WMMSE(H, P_lin, Pn_lin);

        % fix W, optimize over S using fullsearch or fmincon
        % [rate, S_new, ~] = position_optim_MMSE_fmincon(C, S, W, K, M, U, Pn_lin);
        [rate, S_new, ~] = position_optim_MMSE_fullsearch(C, S, W, K, M, U, Pn_lin);

        S = S_new;
        rate_hist(ao_iter) = rate;

        % check convergence
        if ao_iter > 1 && abs(rate_hist(ao_iter) - rate_hist(ao_iter-1)) < tol
            rate_hist = rate_hist(1:ao_iter);
            break;
        end
    end

    % solve for W again with the converged S
    H_opt = build_H_matrix(S, C, K, U);
    [W_opt, R_hist_final, user_rates_final] = precoding_optim_WMMSE(H_opt, P_lin, Pn_lin);

    S_opt = S;
    rate_opt = R_hist_final(end);
end


function H = build_H_matrix(S, C, K, U)
    H = zeros(K, U);
    for u = 1:U
        H(:,u) = sum(S .* C(:,:,u), 2);
    end
end
