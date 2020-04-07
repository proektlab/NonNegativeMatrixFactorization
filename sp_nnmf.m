function [U,V, p] = sp_nnmf( A, k, l, conv, maxiter )
% Simple sparse NNMF with L1 norm penalty for sparsity. Adapted from
% http://arxiv.org/pdf/1305.7169.pdf algorithm 1. INPUT: A is the matrix to
% be factorized, k is the numbe of components to obtain, conv is the
% threshold for convergence (as fraction improvement in performance,
% maxiter is the number of iterations. OUTPUTS: U and V are factors such
% that A =U*V' and p is the sequence of model goodness of fit according to
% the penalty. 

if nargin==2;
    l=0.1;                  % default sparsity penalty
    conv=0.00001;           % default threshold for convergence
    maxiter=5000;           % default maximum number of iterations

elseif nargin==3;
    conv=0.00001;
    maxiter=5000;

elseif nargin==4;
    maxiter=5000;

elseif nargin<2
    error('At least 2 input variables required');
end


if min(A(:))<0
    error('All entries in the matrix must be positive for NNMF');
end

if ~rem(k,1)==0
    error('Order of NNMF must be an integer');
end

[n,m]=size(A);

% if k>min(n,m)
%     error('The factorization rank is greater than the original matrix');
% end

% initialize with random matrices
U=rand(n,k);
V=rand(m,k);
% initialize empty performance vector
p=zeros(maxiter,1);
% dummie value to enter the loop
improv=10;
%initialize count of iterations
counter=0;

while abs(improv)>conv || improv<0 
    U=uUpdate(U,V,A);
    V=vUpdate(U,V,A, l);
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