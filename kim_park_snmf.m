function [U, V, p] = kim_park_snmf(A, k, fro_wt, sp_wt, dim_sparse, conv, maxiter, U, V)
% Implements Hyunsoo Kim & Haesun Park's sparse nonnegative matrix factorization
% (SNMF/L or SNMF/R) (https://academic.oup.com/bioinformatics/article/23/12/1495/225472)
% See also https://arxiv.org/pdf/1507.03194.pdf
%
% The loss function minimizes squared reconstruction error, plus the Frobenius norm
% of one of the factors and the sum of squared L1 norms of rows of the other factor.
% This is intended to impose sparsity across modes, but not along time or input
% feature dimensions - e.g. to reduce clustering ambiguity.
%
% Inputs ([] indicates default):
%   A:          the matrix to be factorized (n x m)
%   k:          number of components
%   fro_wt:     [0.1] weight of Frobenius-norm regularization (of U or V)
%   sp_wt:      [0.1] weight of sparse-row regularization (of V or U, respectively)
%   dim_sparse: [2] which dimension of A to decompose using sparse rows (1 = U, 2 = V)
%   conv:       [1e-5] threshold for convergence as fraction improvement in performance
%   maxiter:    [5000] maximum number of iterations
%   U           [[]] if nonempty, fix U and only optimize V
%   V           [[]] if nonempty, fix V and only optimize U
%
% Outputs:
%  U and V: n x k and m x k matrices such that A ~= U * V'
%  p:       sequence of model goodness-of-fit values according to the penalty

assert(nargin >= 2, 'At least A and k are required');

[n, m] = size(A);
assert(min(A(:)) >= 0, 'All entries in the matrix must be nonnegative for NMF');
assert(rem(k, 1) == 0, 'Order of NMF must be an integer');

if nargin < 3 || isempty(fro_wt)
    fro_wt = 0.1;
else
    assert(fro_wt >= 0, 'Frobenius reg weight must be nonnegative');
end

if nargin < 4 || isempty(sp_wt)
    sp_wt = 0.1;
else
    assert(sp_wt >= 0, 'Sparse reg weight must be nonnegative');
end

if nargin < 5 || isempty(dim_sparse)
    dim_sparse = 2;
else
    assert(isscalar(dim_sparse) && any(dim_sparse == [1, 2]), 'Sparse dim must be 1 or 2');
end

if nargin < 6 || isempty(conv)
    conv = 1e-5;
end

if nargin < 7 || isempty(maxiter)
    maxiter = 5000;
end

if nargin >= 8 && ~isempty(U)
    fixU = true;
    assert(ismatrix(U) && all(size(U) == [n, k]), 'Provided U has wrong size');
else
    fixU = false;
    % initialize with random matrix
    U = rand(n, k);
end

if nargin >= 9 && ~isempty(V)
    fixV = true;
    assert(ismatrix(V) && all(size(V) == [m, k]), 'Provided V has wrong size');
    
    if fixU
        warning('Both U and V provided - nothing to do!');
        p = [];
        return;
    end
else
    fixV = false;
    % initialize with random matrix
    V = rand(m, k);
end

% initialize empty performance vector
p = zeros(maxiter, 1);

for kIter = 1:maxiter
    if ~fixU
        U = update_u(U, V, A);
    end
    
    if ~fixV
        V = update_v(U, V, A);
    end
    
    p(kIter) = pen(A, U, V);
    
    if kIter > 1
        improv = (p(kIter-1) - p(kIter)) / p(kIter-1);
        if improv > 0 && improv < conv
            fprintf('Converged in %d iterations\n', kIter);
            break;
        elseif kIter == maxiter
            warning('Maximum number of iterations reached without convergence');
        end
    end
end

% truncate to the actual number of iterations
p = p(1:kIter);

% Below 2 fns are Algo 1 of http://arxiv.org/pdf/1305.7169.pdf
% with modifications for either frobenius or sparse norm regularization.
    function Un = update_u(U, V, A)
        if dim_sparse == 1 % do sparse reg
            V_aug = [V; sqrt(sp_wt) * ones(1, k)];
            A_aug = [A, zeros(n, 1)];
        else % do fro reg
            V_aug = [V; sqrt(fro_wt) * eye(k)];
            A_aug = [A, zeros(n, k)];
        end
        
        num = A_aug * V_aug;
        den = U * (V_aug' * V_aug);
        Un = U .* (num ./ den);
        Un(isnan(Un)) = 0;
    end

    function Vn = update_v(U, V, A)
        if dim_sparse == 2 % do sparse reg
            U_aug = [U; sqrt(sp_wt) * ones(1, k)];
            A_aug = [A; zeros(1, m)];
        else % do fro reg
            U_aug = [U; sqrt(fro_wt) * eye(k)];
            A_aug = [A; zeros(k, m)];
        end
        
        num = A_aug' * U_aug;
        den = V * (U_aug' * U_aug);
        Vn = V .* (num ./ den);
        Vn(isnan(Vn)) = 0;
    end

% loss function (penalty)
    function p = pen(A, U, V)        
        err = norm(A - U*V', 'fro')^2;
        
        if dim_sparse == 1 % SNMF/L
            fro_reg = fro_wt * norm(V, 'fro')^2;
            sp_reg = sp_wt * norm(sum(U, 2))^2;
        else % SNMF/R
            fro_reg = fro_wt * norm(U, 'fro')^2;
            sp_reg = sp_wt * norm(sum(V, 2))^2;
        end
        
        p = err + fro_reg + sp_reg;
    end
end
