function [best_sum_rate, best_S, best_indexes] = position_optim_MMSE_fmincon(C, S, W, K, M, U, Pn_lin)
% function to optimize matrix S to find the optimal PA activation using
% fmincon
%
%   C       : K x M x U effective channel matrix, H(:,u) = h_u
%   W       : K x U precoding matrix
%   K       : number of waveguide 
%   M       : number of possible activation points per waveguide
%   U       : number of users
%   Pn_lin  : noise power in linear scale
%
%   best_sum_rate   : optimal sum rate with best_S matrix
%   best_S          : K x M, optimal matrix S as output of fmincon
%   best_indexes    : K x 1, indexes of activation on each waveguide

    options = optimoptions('fmincon', ...
        'Algorithm', 'interior-point', ...
        'MaxIterations', 1e6, ...
        'MaxFunctionEvaluations', 1e6, ...
        'StepTolerance', 1e-12, ...
        'OptimalityTol', 1e-9, ...
        'Display', 'none');
    
    [Aeq, beq, ~, lb, ub] = constr_init_fmincon(K,M);

    % wrap rate and constraints
    rate = @(S) neg_sum_rate_MMSE_given_S(C,S,W,K,U,Pn_lin);
    nonlcon = @(S) S_constraints(S); 

    best_S = fmincon(rate, S, [], [], Aeq, beq, lb, ub, nonlcon, options);

    [~, best_indexes] = max(best_S, [], 2); % from matrix S to indexes of the maxima

    best_sum_rate = -neg_sum_rate_MMSE_given_S(C,best_S,W,K,U,Pn_lin);

end    


%% function to compute negative sum rate
function neg_rate = neg_sum_rate_MMSE_given_S(C,S,W,K,U,Pn_lin)
    H = zeros(K, U); % build new H form new S and C
    for k = 1:K
        for u = 1:U
            H(k,u) = S(k,:) * C(k,:,u).';
        end
    end

    interf_plus_noise = sum(abs(H'*W).^2, 2) + Pn_lin; % sum_i |h_u^H w_i|^2 + Pn_lin
    e = ones(U,1) - (abs(diag(H'*W)).^2 ./ interf_plus_noise); % Ux1
    v = 1 ./ e;
    neg_rate = -sum(log2(v));
end

%% constraints on S
function [ineqnonlin, eqnonlin] = S_constraints(S)
% Nonlinear constraints, specified as a function handle or function name. 
% nonlcon is a function that accepts a vector or array x and returns two arrays, 
% ineqnonlin(x) and eqnonlin(x).

    ineqnonlin = [];    
    eqnonlin = [];   % Compute nonlinear equalities at x, no inequality constraints in this case
end

%% constraints and initialization for fmincon
function [Aeq, beq, S_0, lb, ub] = constr_init_fmincon(K,M)
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
    ub = ones(size(S_0));
end 