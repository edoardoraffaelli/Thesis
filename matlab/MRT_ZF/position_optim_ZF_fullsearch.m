function [best_sum_rate, best_S, best_indexes] = position_optim_ZF_fullsearch(C, K, M, U, P_lin, Pn_lin)
% function to optimize matrix S to find the optimal PA activation using
% full search across every combination of indexes
%
%   C       : K x M x U effective channel matrix, H(:,u) = h_u
%   K       : number of waveguide 
%   M       : number of possible activation points per waveguide
%   U       : number of users
%   P_lin   : tx power in linear scale
%   Pn_lin  : noise power in linear scale
%
%   best_sum_rate   : optimal sum rate with best_S matrix
%   best_S          : K x M, optimal matrix S as output of fmincon
%   best_indexes    : K x 1, indexes of activation on each waveguide
    
    best_sum_rate = 0;
    best_indexes = ones(K,1);

    % generate all M^K combinations
    grids = cell(1, K); % = {[]  []  []}
    [grids{:}] = ndgrid(1:M); % [1 ... M]

    % Put all combinations into a matrix
    all_indexes = zeros(M^K, K);
    for k = 1:K
        all_indexes(:,k) = grids{k}(:);
    end

    % Exhaustive search
    for i = 1:size(all_indexes, 1)
        indexes_tmp = all_indexes(i,:);
        
        H_ZF = zeros(K, U); % build H from C
        for u = 1:U
            for k_ = 1:K
                H_ZF(k_, u) = C(k_, indexes_tmp(k_), u);
            end
        end

        F_ZF = H_ZF / (H_ZF' * H_ZF);

        user_rate_ZF = zeros(U,1);
        for u = 1:U
            gain = abs(H_ZF(:,u)' * F_ZF(:,u))^2 / norm(F_ZF(:,u))^2;
            user_rate_ZF(u) = log2(1 + gain * (P_lin/U) / Pn_lin);
        end

        sum_rate_tmp = sum(user_rate_ZF(:));            

        if sum_rate_tmp > best_sum_rate  % check if is best rate  
            best_sum_rate = sum_rate_tmp;
            best_indexes = indexes_tmp;
        end
    end

    best_S = zeros(K,M);
    for k = 1:K
        best_S(k, best_indexes(k)) = 1;
    end
end
