function [best_sum_rate, best_S, best_indexes] = position_optim_ZF_fmincon(C, K, M, U, P_lin, Pn_lin)
% function to optimize matrix S to find the optimal PA activation using
% fmincon
%
%   C       : K x M x U effective channel matrix
%   K       : number of waveguide 
%   M       : number of possible activation points per waveguide
%   U       : number of users
%   P_lin   : tx power in linear scale
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
    
    [Aeq, beq, S_0, lb, ub] = constr_init_fmincon(K,M);

    % wrap rate and constraints
    rate = @(S) neg_ZF_sum_rate_given_S(S, C, K, U, P_lin, Pn_lin);
    nonlcon = @(S) S_constraints(S); 

    best_S = fmincon(rate, S_0, [], [], Aeq, beq, lb, ub, nonlcon, options);
    [~, best_indexes] = max(best_S, [], 2); % from matrix S to indexes of the maxima

    best_sum_rate = -neg_ZF_sum_rate_given_S(best_S, C, K, U, P_lin, Pn_lin);
end 


%% function to compute negative sum rate
function neg_sum_rate = neg_ZF_sum_rate_given_S(S, C, K, U, P_lin, Pn_lin)
    H_ZF = zeros(K, U); % build H_ZF form S and C
    for k = 1:K
        for u = 1:U
            H_ZF(k,u) = S(k,:) * C(k,:,u).';
        end
    end
    
    F_ZF = H_ZF / (H_ZF' * H_ZF);
    
    user_rate = zeros(U,1);
    for u = 1:U
        gain = abs(H_ZF(:,u)' * F_ZF(:,u))^2 / norm(F_ZF(:,u))^2;
        user_rate(u) = log2(1 + (P_lin/U) / Pn_lin * gain);
    end
    neg_sum_rate = -sum(user_rate); % negative sum rate
end

%% constraints on S
function [ineqnonlin, eqnonlin] = S_constraints(S)
    % Nonlinear constraints, specified as a function handle or function name. 
    % nonlcon is a function that accepts a vector or array x and returns two arrays, 
    % ineqnonlin(x) and eqnonlin(x).
    
    ineqnonlin = [];    
    eqnonlin = [];   % Compute nonlinear equalities at x, no inequality constraints in this case
    % eqnonlin = sum(S - S.^2, "all");
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