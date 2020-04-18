function  BCV = nnmf_k_ver( A, num, sets, kmin, kmax, conv, maxiter)
%Bi-cross validation of rank of NNMF according to the Algorithm 3 of http://arxiv.org/pdf/0908.2062v1.pdf
%Inputs: A is the original matrix to be factorized, num is a 2 element
%vector that contains the number of rows and columns to hold out of the
%original matrix for verification purposes. sets is how many different
%unique sets of hold outs to perform, kmin is the minimum rank kmax is the
%maximum rank. 
%Outputs: BCV is a matrix of dimensions setsXlength(kmin:kmax) where each
%column contains the reconstruction error for a given value of k and each row is each individual set.

if nargin < 6 || isempty(conv)
    conv = 1e-5;
end

if nargin < 7 || isempty(maxiter)
    maxiter = 5000;
end

[n,m]=size(A);
rowHold=zeros(num(1), sets);
colHold=zeros(num(2), sets);

% create hold out sets
for i=1:sets
    rowHold(:,i)=randsample(n, num(1));
    colHold(:,i)=randsample(m, num(2));
end
% create k values
k=kmin:kmax;
% initialize the array for validation 
BCV=zeros(sets,length(k));

% loop over sets;
for j=1:sets
    temp=A;
    temp(rowHold(:,j),:)=[];        % matrix with rows removed
    A_r=temp;

    temp=A;
    temp(:,colHold(:,j))=[];
    A_c=temp;                       % matrix with columns removed

    temp=A_r;
    temp(:,colHold(:,j))=[];        % matrix with both rows and columns removed
    A_rc=temp;
% parallel loop for all values of K
    parfor i=1:length(k)
        % algo 3 in http://arxiv.org/pdf/0908.2062v1.pdf
        [U_tilde, V_tilde, ~]=sp_nnmf(A_rc,k(i), 0, conv, maxiter); % regular NNMF. sparsity constraint removed
        U_hat=sp_nnmf(A_c, k(i), 0, conv, maxiter/10, [], V_tilde); % NNMF holding V constant
        [~, V_hat]=sp_nnmf(A_r, k(i), 0, conv, maxiter/10, U_tilde);% NNMF holding U constant
        BCV(j, i)=pen(A, U_hat, V_hat);                         % Frobenius norm of the difference between full and K-ranked reconstruction
    end

end


end

function p=pen(A,U,V )
% penalty that includes the L1 norm for sparsity;
p=norm(A-U*V', 'fro');

end