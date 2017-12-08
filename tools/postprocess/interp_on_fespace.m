function [interpsol] = interp_on_fespace(fespace1,sol1,target_fespace)

nodes = target_fespace.nodes;

interpsol = zeros(size(nodes,1),1);

for i = 1:length(nodes)
    node = nodes(i,1:2);
    
    [I1,code1] = interpolate_in_point(fespace1,sol1,node(1),node(2));
    
    if (code1 == 0)
        interpsol(i) = I1;
    else
        error(['Point (',num2str(node(1)),',',num2str(node(2)),') is outside the domain']);
    end
end