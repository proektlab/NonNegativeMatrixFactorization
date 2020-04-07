function [ U,V, p ] = sp_sm_nnmf(A,k, lt, ls, W, conv, maxiter  )
% This performs sprase NNMF with temporal smoothing as in Algo 2. in http://arxiv.org/pdf/1305.7169.pdf
% A is the matrix to be factorized (3D where the third dimension is time).
% K is the rank of the desired matrix
% lt is the weight for the temporal smoothing, ls is the weight for
% sparsity. W is the length in terms of units of the third dimension of A
% for the window along which the smoothness in time is calculated. conv is
% the parameter for convergence (decrease in goodness of fit for each
% successive iteration) maxiter is the maximum number of iterations.

%Outputs: U and V are going to be the factors such that
%A(:,:,t)=U(:,:,t)*V(:,:, t)'. If A is n by q by t  then U is n by k by t and V is q by k by t. p is the performance on each successive
%iteration where length of p is the number of iterations taken to
%conversion. 


if length(size(A))~=3
    error('The input Matrix must be 3 dimensional where the 3rd dimension is time');
else
    [n,m,T]=size(A);
end

% check for reasonable input variables
if min(A(:))<0
    error('All entries in the matrix must be positive');    
elseif rem(k,1)~=0
    error('Rank for NNMF must be an integer');
elseif rem(W,1)~=0
    error('Time window for temporal smoothing must be an integer');
elseif k>min(m,n)
    error('Rank for NNMF should be lower than the rank of the original matrix');
end

% set default values if not supplied
if nargin <2
    error('At least 2 input variables required');
end
if nargin==3;
    lt=0.1;
    ls=0.1;
    W=2;
    conv=0.000001;           % default threshold for convergence
    maxiter=5000;           % default maximum number of iterations  
end

if nargin==4
    ls=0.1;
    W=2;
    conv=0.000001;           % default threshold for convergence
    maxiter=5000;           % default maximum number of iterations  
end

if nargin==5
    W=2;
    conv=0.000001;           % default threshold for convergence
    maxiter=5000;           % default maximum number of iterations  
end

if nargin==6
    conv=0.000001;           % default threshold for convergence
    maxiter=5000;           % default maximum number of iterations 

end

if nargin==7
    maxiter=5000;           % default maximum number of iterations 
end

U=rand(n,k, T);
V=rand(m,k,T);


p=zeros(maxiter,1);
improv=10;
counter=0;

while abs(improv)>conv || improv<0 
    for i=1:T
        U(:,:,i)=uUpdate(A,V,U, lt, W, i);
        V(:,:,i)=vUpdate(A(:,:,i),V(:,:,i),U(:,:,i), ls);
    end

    p(counter+1)=penalty(A, V, U, lt, ls, W);
    
    if counter>=1
        improv=(p(counter)-p(counter+1))./p(counter);
    end
    counter=counter+1;
    if counter>=maxiter
        disp('Maximum number of iterations reached without convergence');
        improv=0;   % kill the loop
    end
end
p=p(1:counter);
end

% Update functions according to Algorithm 2 http://arxiv.org/pdf/1305.7169.pdf

function Un=uUpdate(A, V, U, lt, W,  t)
    Upres=U(:,:, t);
        
    % weighted sum over U's across time (W determines the temporal window)
    past=lt*sum(U(:,:,max(1, t-W/2):t-1),3);
    future=lt*sum(U(:,:,t+1:min(size(U,3), t+W/2)),3);
    
    
    num=A(:,:,t)*V(:,:,t)+past+future;
    den=Upres*(V(:,:,t)'*V(:,:,t))+W*lt*Upres;
    Un=Upres.*(num./den);




end

function Vn=vUpdate(A,V,U, l)
% same as for sparse NNMF 
num=A'*U;
den=V*(U'*U)+l;
Vn=V.*(num./den);

end


function p=penalty(A, V, U, lt, ls, W)
% penalty computed according to http://arxiv.org/pdf/1305.7169.pdf Eq(9);
tP=zeros(size(U,3),1);
% compute the penalty for non-smoothess for each element T
for i=1:length(tP)
   tP(i)=TimePenalty(U, W, i); 
end

%sum of non-smoothness penalties over time weighed by constant
tP=lt*sum(tP);

fro=zeros(size(U,3),1);

% Frobenius Norm at each time point
for i=1:length(fro)
   fro(i)=norm(A(:,:,i)-U(:,:,i)*V(:,:,i)', 'fro');    
end
%sum over all time;
fro=sum(fro);

% L1 sparsity penalty for each time point

sp=zeros(size(U,3),1);

for i=1:length(sp);
   sp(i)=sum(sum(V(:,:,i),1));     
end
sp=ls*sum(sp);


% combined penalty is the sum of Frobenius reconstruction error,
% smoothness, and sparsity constraints. 
p=fro+tP+sp;



end

function pt=TimePenalty(U, W, t)

%initialize the matrix for Frobenius Norm
f=zeros(length(max(t-W/2, 1):min(size(U,3), t+W/2))-1, 1);

% sum the frobenius norm for the differences between Ut and each element of
% the window length W centered at t.
for i=1:length(f)
   f(i)=norm(U(:,:,t)-U(:,:,(t-W/2)+i), 'fro'); 

end

pt=sum(f);
end