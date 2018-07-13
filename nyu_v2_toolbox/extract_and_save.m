function extract_and_save()
    % data = load('nyu_depth_v2_labeled.mat');
    data_dir = './data/nyu_depth_v2_labeled/';
    rgb_dir = strcat(data_dir, 'rgb');
    depth_dir = strcat(data_dir, 'depth');
    display(rgb_dir);
    display(depth_dir);
    for i = 1:length(data.scenes)
        rgb = data.images(:,:,:,i);
        depth = data.depths(:,:,i);
        rgb_file = [rgb_dir '/' num2str(i) '.jpg'];
        depth_file = [depth_dir '/' num2str(i) '.mat'];
        imwrite(rgb, rgb_file);
        save(depth_file, 'depth');
        
        fprintf('Saved to %s and %s\n', rgb_file, depth_file);
    end
end