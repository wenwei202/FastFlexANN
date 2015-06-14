function [ mat ] = normalization_column( mat,mode )
    [row,col] = size(mat);
    if strcmp(mode,'minmax')
        mat = (mat-repmat(min(mat),row,1))./repmat(max(mat)-min(mat),row,1);
    else
        if strcmp(mode,'meanvar')
            mat = (mat-repmat(mean(mat),row,1))./repmat(sqrt(var(mat)),row,1);
        else
            error('wrong mode');
        end
    end

end

