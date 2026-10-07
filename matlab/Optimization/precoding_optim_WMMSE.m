function [W, R_hist, user_rates_final] = precoding_optim_WMMSE(H, P_lin, Pn_lin, max_iter, tol)
% WMMSE  Weighted-MMSE sum-rate beamforming under a total power constraint.
%
%   H      : K x U effective channel matrix, H(:,u) = h_u
%   P_lin  : total transmit power budget, sum_u ||w_u||^2 <= P
%   Pn_lin : noise power
%   max_iter : max number of outer WMMSE iterations
%   tol    : convergence tolerance
%
%   W       : K x U beamforming matrix
%   R_hist  : vector of sum-rate values at each iteration
%   user_rates : individual user rates


    if nargin < 5, max_iter = 100; end
    if nargin < 6, tol = 1e-6;    end

    [K, U] = size(H);

    % initialization: MRT direction
    W = H;
    W = W ./ norm(W, 'fro') * sqrt(P_lin); % scale so sum_u ||w_u||^2 = P

    R_hist = zeros(max_iter, 1);

    for iter = 1:max_iter

        % MMSE receiver update
        interf_plus_noise = sum(abs(H'*W).^2, 2) + Pn_lin; % sum_i |h_u^H w_i|^2 + Pn_lin
        a = diag(W'*H) ./ interf_plus_noise;    % Ux1, MMSE receiver

        % MMSE weight update
        % e_u = 1 - u_u^ * h_u^H w_u  
        % v_u = 1/e_u = 1 + gamma_u
        e = ones(U,1) - (abs(diag(H'*W)).^2 ./ interf_plus_noise); % Ux1
        v = 1 ./ e;
    
        % add to sum-rate history
        R_hist(iter) = sum(log2(v));

        % precoder update, bisection on lambda
        W = update_W(H, a, v, P_lin, K, U);

        % check convergence
        if iter > 1 && abs(R_hist(iter) - R_hist(iter-1)) < tol
            R_hist = R_hist(1:iter);
            break;
        end
    end

    % compute final user rates
    interf_plus_noise = sum(abs(H'*W).^2, 2) + Pn_lin;
    e = ones(U,1) - (abs(diag(H'*W)).^2 ./ interf_plus_noise);
    user_rates_final = -log2(e);
end


function W = update_W(H, a, v, P_lin, K, U)
    % build the shared matrix  M(lambda) = sum_i v_i*|a_i|^2 h_i h_i^H + lambda*I
    weighted_sum = zeros(K, K);
    for i = 1:U
        hi = H(:, i);
        weighted_sum = weighted_sum + v(i) * abs(a(i))^2 * (hi * hi');
    end

    compute_W = @(lambda) compute_W_given_lambda(H, a, v, weighted_sum, lambda, K, U);

    % try lambda = 0
    W0 = compute_W_given_lambda(H, a, v, weighted_sum, 0, K, U);

    if norm(W0, "fro")^2 <= P_lin
        W = W0;
        return;
    end

    % bisect on lambda > 0 until sum_u ||w_u(lambda)||^2 = P
    power_W = @(lambda) norm(compute_W(lambda), "fro")^2 - P_lin;

    lambda_low = 0;
    lambda_high = 1;
    while power_W(lambda_high) > 0  % try lam=1, grow upper bracket until power <= P
        lambda_high = lambda_high * 2;
        if lambda_high > 1e12  % safety cap
            break;
        end
    end

    for bisection_iter = 1:60
        lambda_mid = 0.5 * (lambda_low + lambda_high);
        if power_W(lambda_mid) > 0
            lambda_low = lambda_mid;
        else
            lambda_high = lambda_mid;
        end
    end

    W = compute_W(0.5 * (lambda_low + lambda_high));
end


function W = compute_W_given_lambda(H, a, v, weighted_sum, lambda, K, U)
    M = weighted_sum + lambda * eye(K);
    W = zeros(K, U);

    if lambda == 0
        for u = 1:U
            hu = H(:, u);
            W(:, u) = v(u) * conj(a(u)) * (pinv(M) * hu); 
        end
    else
        for u = 1:U
            hu = H(:, u);
            W(:, u) = v(u) * conj(a(u)) * (M \ hu);
        end
    end
end
