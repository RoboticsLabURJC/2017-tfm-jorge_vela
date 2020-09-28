
TOOLBOX_BADACOST_PATH = '/home/jmbuena/matlab/toolbox.badacost';
% KITTI_PATH = '/home/imagenes/CARS_DATABASES/KITTI_DATABASE/';
% OUTPUT_DATA_PATH =  '../KITTI_CARS_DETECTION_EXPERIMENTS';
% PREPARED_DATA_PATH =  'KITTI_TRAINING_DATA';
% 
% SHRINKAGE = 0.05;
% FRAC_FEATURES = 1/32;
% D = 8;
% T = 1024;
% N = 7500;
% NA = 30000;
% costsAlpha = 1;
% costsBeta = 3;
% costsGamma = 3;
% 
% TRAINED_DETECTOR_FILE = fullfile(OUTPUT_DATA_PATH, ...
%                                  sprintf('BADACOST_%d_%d_%d_D_%d_T_%d_N_%d_NA_%d_S_%4.4f_F_%4.4f', costsAlpha, costsBeta, costsGamma, D, T, N, NA, SHRINKAGE, FRAC_FEATURES), ...
%                                  'KITTI_SHRINKAGE_0.050000_RESAMPLING_1.000000_ASPECT_RATIO_1.750000_Detector.mat');
% det = load(TRAINED_DETECTOR_FILE);


% Test for real image: 
img_name = 'index.jpeg';
I = imread(img_name);
sz = size(I);

scale = 0.99;
fprintf('Test scale = %2.4f\n', scale);
method = 'bilinear';
norm = 1.0;
Iresampled = imResample(I, scale, 'bilinear', 1.0 );
figure; imshow(Iresampled);
save_imResample_results(Iresampled, scale, method, norm, 'index1_imResample_scale_0_99_method_bilinear_norm_1.yaml');

scale = 0.85;
fprintf('Test scale = %2.4f\n', scale);
method = 'bilinear';
norm = 1.0;
Iresampled = imResample(I, scale, 'bilinear', 1.0 );
figure; imshow(Iresampled);
save_imResample_results(Iresampled, scale, method, norm, 'index1_imResample_scale_0_85_method_bilinear_norm_1.yaml');

scale = 0.57;
fprintf('Test scale = %2.4f\n', scale);
method = 'bilinear';
norm = 1.0;
Iresampled = imResample(I, scale, 'bilinear', 1.0 );
figure; imshow(Iresampled);
save_imResample_results(Iresampled, scale, method, norm, 'index1_imResample_scale_0_57_method_bilinear_norm_1.yaml');

scale = 0.5;
fprintf('Test scale = %2.4f\n', scale);
method = 'bilinear';
norm = 1.0;
Iresampled = imResample(I, scale, 'bilinear', 1.0 );
figure; imshow(Iresampled);
save_imResample_results(Iresampled, scale, method, norm, 'index1_imResample_scale_0_5_method_bilinear_norm_1.yaml');

scale = 0.38;
fprintf('Test scale = %2.4f\n', scale);
method = 'bilinear';
norm = 1.0;
Iresampled = imResample(I, scale, 'bilinear', 1.0 );
figure; imshow(Iresampled);
save_imResample_results(Iresampled, scale, method, norm, 'index1_imResample_scale_0_38_method_bilinear_norm_1.yaml');

