function res = MNISTParser(filename)
    fp = fopen(filename,'r');
    magic_number = fread(fp,1,'uint32',0,'b');
    items_number = fread(fp,1,'uint32',0,'b');
    if 2049==magic_number
        res = fread(fp,items_number,'uint8',0,'b');
    else
        if 2051==magic_number
            img_rows = fread(fp,1,'uint32',0,'b');
            img_cols = fread(fp,1,'uint32',0,'b');
            res = zeros(items_number,img_rows,img_cols);
            for i=1:items_number
                res(i,:,:) = fread(fp,[img_cols,img_rows],'uint8',0,'b')';
            end
        else
            error('wrong magic number');
        end
    end
    fclose(fp);
    
end