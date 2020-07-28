function [U,V, p] = sp_nnmf( A, k, l, conv, maxiter, U, V)
% Simple sparse NNMF with L1 norm penalty for sparsity. Adapted from
% http://arxiv.org/pdf/1305.7169.pdf algorithm 1.
%
% INPUT: A is the matrix to
% be factorized, k is the numbe of components to obtain, conv is the
% threshold for convergence (as fraction improvement in performance,
% maxiter is the number of iterations. If U or V is provided, that
% matrix is fixed and only the other is optimized.
%
% OUTPUTS: U and V are factors such
% that A =U*V' and p is the sequence of model goodness of fit according to
% the penalty.

assert(nargin >= 2, 'At least 2 input variables required');

[n,m]=size(A);

if min(A(:))<0
    error('All entries in the matrix must be positive for NNMF');
end

if ~rem(k,1)==0
    error('Order of NNMF must be an integer');
end

if nargin < 3 || isempty(l)
    l=0.1;                  % default sparsity penalty
end

if nargin < 4 || isempty(conv)
    conv=0.00001;           % default threshold for convergence
end

if nargin < 5 || isempty(maxiter)
    maxiter=5000;           % default maximum number of iterations
end

if nargin >= 6 && ~isempty(U)
    fixU = true;
    assert(ismatrix(U) && all(size(U) == [n, k]), 'Provided U has wrong size');
else
    fixU = false;
    % initialize with random matrix
    U = rand(n, k);
end

if nargin >= 7 && ~isempty(V)
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

% if k>min(n,m)
%     error('The factorization rank is greater than the original matrix');
% end

% initialize empty performance vector
p=zeros(maxiter,1);
% dummie value to enter the loop
improv=10;
%initialize count of iterations
counter=0;

while abs(improv)>conv || improv<0
    if ~fixU
        U=uUpdate(U,V,A);
    end

    if ~fixV
        V=vUpdate(U,V,A, l);
    end

    p(counter+1)=pen(A, U, V, l);
    
    if counter>=1
        improv=(p(counter)-p(counter+1))./p(counter);
    end
    counter=counter+1;
    if counter>=maxiter
        disp('Maximum number of iterations reached without convergence');
        improv=0;   % kill the loop
    end

end
% truncate to the actual number of iterations
p=p(1:counter);
end


function Un=uUpdate(U,V, A)
% as in Algo 1 of the paper
num=A*V;
den=U*(V'*V);
Un=U.*(num./den);
Un(isnan(Un))=0;
end

function Vn=vUpdate(U,V,A, l)
% as in Algo 1 of the paper
num=A'*U;
den=V*(U'*U)+l;
Vn=V.*(num./den);
Vn(isnan(Vn))=0;

end

function p=pen(A,U,V, l)
% penalty that includes the L1 norm for sparsity;
p=norm(A-U*V', 'fro')+l*sum(sum(V,1));

end