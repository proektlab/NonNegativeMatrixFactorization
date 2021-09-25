function [X] = symnnmf(G,k, upper_t, norm_prob)
%symmetric NNMF using an alogirthm due to Wang, Fei, et al. "Community discovery using nonnegative matrix factorization." Data Mining and Knowledge Discovery 22.3 (2011): 493-521.
% G is a symmetric scquare matrix which denotes adjacency between points
% (e.g. 1 is connected and zero is disconnected). k is the number of
% desired cluster. The output is a matrix X (size(G), k). Each column of X corresponds
% to a cluster. Each element in this column corresponds to the probability
% that each element belongs to a particular cluster. 
%  
% Updated 09/21 DGD 
% Updated to allow for option to decide wether to only evaluate the upper
% triangle of a matrix to assess goodness of fit (boolean input upper_t).
% Also updated to allow option to normalize elements as as probability (boolean input norm_prob)
 
maxiter=50000; % Icreased max iterations cap DGD 09/21
Ftol=0.00000001;
 
% check if square
[S1, S2]=size(G);
 
if S1~=S2
    error('Graph needs to be square');
end
 
% check for symmetry
if any(G~=G', 'all')
    error('This algorithm works only on symmetric graphs');
end
 
% intial guess
X=rand(S1,k);
i=1;
F=1000;
c=1000;
while i<=maxiter && F>Ftol
   X_new=updateF(G,X);
   c_new=cost(G,X_new, upper_t);
   F=abs(c_new-c)./c;
   c=c_new;
   X=X_new;
   i=i+1;
end
if i==maxiter
   disp('Maximum Iterations reached without convergence'); 
else
    disp(['Converged after ' num2str(i) ' iterations']);
end
 
% reorder the columns in terms of goodness of reconstruction;
Ord=zeros(k,1);
for i=1:length(Ord)
   Ord(i)=cost(G,X(:,i), upper_t); 
end
[~,Ind]=sort(Ord, 'ascend');
X=X(:,Ind);
 
% option to do the normalization DGD 09/21
if (norm_prob)
    X=X./sum(X,2);                  % normalize such that the elements denote probabilities.
end
 
 
    
    function Xnew=updateF(g, X_old)                              % update rule according to https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&ved=2ahUKEwiI6v2siNfoAhXLVN8KHbPJDd0QFjABegQIBxAB&url=https%3A%2F%2Fusers.cs.fiu.edu%2Farchive%2Ftaoli%2Fpub%2Fdmkd-community-discovery.pdf&usg=AOvVaw3MqztnYJb8-zh_5uAaQ08d
        Xnew=X_old.*(1/2+(g*X_old)./((2.*X_old*X_old')*X_old));
    end
 
    function C=cost(G, X, upper_t)   % cost function to be optimized
        
        if (upper_t)
            C=norm(triu(G-X*X', 1), 'fro'); % option to only consider the upper triangular portion DGD 09/21 
        else
            C=norm(G-X*X', 'fro');
        end
        
    end
end
 
