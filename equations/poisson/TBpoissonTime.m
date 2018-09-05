function [sol] = TBpoissonTime(param,FOM)
    % TBPOISSON

    %   Author: Stefano Pagani <stefano.pagani at polimi.it>

    ntimestep = FOM.T./FOM.dt;

    nupdate = ntimestep/FOM.Ntparam;
    
    % Solver
    sol = repmat(FOM.uL, 1,  ntimestep);
    
    A = param( 1 ,1 )*FOM.A_in{1};
    for i=2:size(param,2)
        A = A + param( 1 ,i)*FOM.A_in{i};
    end

    b = FOM.b;
    
    % intial condition
    sol(FOM.iN,1)  = A\b;
    
    size(param)
    
    for iT = 1:ntimestep

        ceil(iT/nupdate)
        % 
        A = param( ceil(iT/nupdate) ,1 )*FOM.A_in{1};
    %     if (nargin==3)
    %         b = b - param(1)*A_in_mean{1};
    %     end

        for i=2:size(param,2)
            A = A + param( ceil(iT/nupdate) ,i)*FOM.A_in{i};

    %         if (nargin==3)
    %             b = b - param(i)*A_in_mean{i};
    %         end
        end

        sol(FOM.iN,iT+1)  = (FOM.M + FOM.dt* A ) \ ( FOM.M*sol(FOM.iN,iT) + FOM.dt*b ) ;
        
            
        plot_fe_function(sol(:,iT+1),FOM.fespace)
        axis( [0 1.5 0 1.5 0 7] )
        
        pause(1)
    
    end
end

