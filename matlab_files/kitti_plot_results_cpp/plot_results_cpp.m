
% Set here the BAdaCost matlab toolbox for detection (modification from P.Dollar's one):
TOOLBOX_BADACOST_PATH = '/home/jmbuena/matlab/toolbox.badacost';
%KITTI_PATH = '/home/imagenes/CARS_DATABASES/KITTI_DATABASE/';
OUTPUT_DATA_PATH =  'KITTI_CARS_DETECTION_EXPERIMENTS';
PREPARED_DATA_PATH =  'KITTI_TRAINING_DATA';

% ------------------------------------------------------------------------
% Plot experiments on SAMME vs BAdaCost vs SubCat
clear results_dirs;
clear legend_text;
i = 1;
results_dirs{i} = fullfile(OUTPUT_DATA_PATH, 'BADACOST_1_3_3_D_8_T_1024_N_7500_NA_30000');
legend_text{i} =  'BAdaCost Matlab';
i = i + 1;
results_dirs{i} = fullfile(OUTPUT_DATA_PATH, 'BADACOST_CPP');
legend_text{i} =  'BAdaCost C++';
i = i + 1;
% results_dirs{i} = fullfile(OUTPUT_DATA_PATH, 'SAMME_D_8_T_1024_N_7500_NA_30000');
% legend_text{i} =  'SAMME, D=8';
% i = i + 1;
fig_filename = 'FIGURE_BADACOST_PAPER_VS_CPP';
plot_result_fig_kitti(PREPARED_DATA_PATH, OUTPUT_DATA_PATH, results_dirs, legend_text, fig_filename);
 
