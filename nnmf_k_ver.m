function  BCV = nnmf_k_ver( A, num, sets, kmin, kmax)
%Bi-cross validation of rank of NNMF according to the Algorithm 3 of http://arxiv.org/pdf/0908.2062v1.pdf
%Inputs: A is the original matrix to be factorized, num is a 2 element
%vector that contains the number of rows and columns to hold out of the
%original matrix for verification purposes. sets is how many different
%unique sets of hold outs to perform, kmin is the minimum rank kmax is the
%maximum rank. 
%Outputs: BCV is a matrix of dimensions setsXlength(kmin:kmax) where each
%column contains the reconstruction error for a given value of k and each row is each individual set.

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
        [U_tilde, V_tilde, ~]=sp_nnmf(A_rc,k(i), 0);            % regular NNMF. sparsity constraint removed
        U_hat=U_onlyNMF(A_c, V_tilde, k(i));                    % NNMF holding V constant
        V_hat=V_only_NMF(A_r, U_tilde, k(i));                   % NNMF holding U constant
        BCV(j, i)=pen(A, U_hat, V_hat);                         % Frobenius norm of the difference between full and K-ranked reconstruction
    end

end


end

function U=U_onlyNMF(A,V, k)
maxiter=500;
conv=0.00001; 
p=zeros(maxiter,1);

[n, m]=size(A);

% initialize U
U=rand(n,k);
improv=10;
% dummie value to enter the loop
improv=10;
%initialize count of iterations
counter=0;

while abs(improv)>conv || improv<0 
    U=uUpdate(U,V,A);
    p(counter+1)=pen(A, U, V); 
    if counter>=1
        improv=(p(counter)-p(counter+1))./p(counter);
    end
    counter=counter+1;
    if counter>=maxiter
     %   disp('Maximum number of iterations reached without convergence');
        improv=0;   % kill the loop
    end

end
% truncate to the actual number of iterations
p=p(1:counter);
end


function V=V_only_NMF(A, U, k)
maxiter=500;
conv=0.00001; 
p=zeros(maxiter,1);

[~, m]=size(A);

% initialize U
V=rand(m,k);
improv=10;
% dummie value to enter the loop
improv=10;
%initialize count of iterations
counter=0;
while abs(improv)>conv || improv<0 
    V=vUpdate(U,V,A);
    p(counter+1)=pen(A, U, V); 
    if counter>=1
        improv=(p(counter)-p(counter+1))./p(counter);
    end
    counter=counter+1;
    if counter>=maxiter
       % disp('Maximum number of iterations reached without convergence');
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

function Vn=vUpdate(U,V,A)
% as in Algo 1 of the paper
num=A'*U;
den=V*(U'*U);
Vn=V.*(num./den);
Vn(isnan(Vn))=0;

end

function p=pen(A,U,V )
% penalty that includes the L1 norm for sparsity;
p=norm(A-U*V', 'fro');

end