function [best_sum_rate, best_S, best_indexes] = position_optim_MMSE_fullsearch(C,S,W,K,M,U,Pn_lin)
% function to optimize matrix S to find the optimal PA activation using
% full search across every combination of indexes
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

    best_sum_rate = 0;
    best_indexes = ones(K,1);
    best_S = S;
    
    grids = cell(1, K);  % generate all M^K combinations
    [grids{:}] = ndgrid(1:M);
    
    all_indexes = zeros(M^K, K);  % all combinations into a matrix
    for k = 1:K
        all_indexes(:,k) = grids{k}(:);
    end

    % exhaustive search
    for i = 1:size(all_indexes, 1) % for all the combination of indexes
        indexes_tmp = all_indexes(i,:);

        S_tmp = zeros(K,M); % build the new S matrix
        for k = 1:K
            S_tmp(k,indexes_tmp(k)) = 1;
        end

        H = zeros(K, U); % build new H form new S and C
        for k = 1:K
            for u = 1:U
                H(k,u) = S_tmp(k,:) * C(k,:,u).';
            end
        end
          
        sum_rate_tmp = compute_sum_rate_MMSE(H,W,U,Pn_lin);

        if sum_rate_tmp > best_sum_rate  % check if it is best rate  
            best_sum_rate = sum_rate_tmp;
            best_indexes = indexes_tmp;
            best_S = S_tmp;
        end
    end
end


function sum_rate = compute_sum_rate_MMSE(H,W,U,Pn_lin)
    interf_plus_noise = sum(abs(H'*W).^2, 2) + Pn_lin; % sum_i |h_u^H w_i|^2 + Pn_lin
    e = ones(U,1) - (abs(diag(H'*W)).^2 ./ interf_plus_noise); % Ux1
    user_rates = -log2(e);
    sum_rate = sum(user_rates);
end

