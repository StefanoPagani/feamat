function [U, V, sigma] = POD(S,tolP,Hnorm)
            %POD proper orthogonal decomposition algorithm
            
            % svd method
            if nargin==2
                [U,Sigma,V] = svd(S,'econ');
            else
                [Low, flag, prm] = chol(Hnorm,'lower','vector');
                S               = Low'*S(prm,:);
                invp            = 0*prm;
                invp(prm)       = 1:length(prm);
                S               = S(invp,:);               
                [U, Sigma, V]   = svd(S,'econ');
                U               = Low' \ U(prm,:);
                U               = U(invp,:);
            end

            % singular values
            sigma = diag(Sigma);

            % truncation
            if tolP < 1
                sigmaC = cumsum(sigma.^2);
                [val,ind] = max( tolP^2>=(1-sigmaC./sigmaC(end)) ) ;
            else
                ind = min(tolP,size(V,2));
            end

            % final basis functions
            V = V(:,1:ind);
            U = U(:,1:ind);
                
            
end
