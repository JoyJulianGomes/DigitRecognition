function showImage(fig400, reshapeData)
    res = reshape(reshapeData, 20, 20, 47, 10);
    for j=1:10%numLabel
        for k=1:47%numImages
            fprintf('RSP ImageNo: %d, Label: %d\n', k, j);

            fig400.set('CurrentObject', imshow(res(:, :, k, j)),...
                         'Position', [1235 180 300 200]);
            pause;
        end
    end
end