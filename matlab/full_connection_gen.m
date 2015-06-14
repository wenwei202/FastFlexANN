function [pair] = full_connection_gen( layers )

    layer_num = length(layers);
    connection_num = 0;
    for i=1:layer_num-1
        connection_num = connection_num + (layers(i)+1)*layers(i+1);
    end
    pair = zeros(connection_num,2);
    idx = 1;
    offset = 0;
    for i=1:layer_num-1
        for m=0:layers(i) %include bias
            for n=0:layers(i+1)-1 %exclude bias
                pair(idx,1) = offset + m;
                pair(idx,2) = offset + layers(i) + 1 + n;
                idx = idx + 1;
            end
        end
        offset = offset + layers(i)+1;
    end
%% write to file
    filestr = sprintf('full_nn');
    for i=1:layer_num
        filestr = strcat(filestr,sprintf('_%d',layers(i)));
    end
    filestr = strcat(filestr,sprintf('.txt'));
    
    fp = fopen(filestr,'w');
    fprintf(fp,'%d ',layer_num);
    for i=1:layer_num
        fprintf(fp,'%d ',layers(i));
    end
    fprintf(fp,'%d %d\n',connection_num,0);

    for i=1:connection_num
        fprintf(fp,'%d %d\n',pair(i,1),pair(i,2));
    end
    fclose(fp);
    connection_num
    
end

