function [sol] = TBpoisson(param,A_in,b,uL,iN,A_in_mean)
    % TBPOISSON implements the solution of the Poisson thermal block
    % problem

    %   Author: Stefano Pagani <stefano.pagani at polimi.it>


    % Solver
    sol = uL;

    A = param(1)*A_in{1};
    if (nargin==6)
        b = b - param(1)*A_in_mean{1};
    end
    
    for i=2:length(param)
        A = A + param(i)*A_in{i};
        
        if (nargin==6)
            b = b - param(i)*A_in_mean{i};
        end
    end

    sol(iN)  = A\b;


end

