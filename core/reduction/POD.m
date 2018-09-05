function [U, V, sigma] = POD(S,tolP)
            %POD proper orthogonal decomposition algorithm

            % svd method
            [U,Sigma,V] = svd(S,'econ');

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
