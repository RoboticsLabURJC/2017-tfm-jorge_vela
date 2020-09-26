
TOOLBOX_BADACOST_PATH = '/home/jmbuena/matlab/toolbox.badacost';
KITTI_PATH = '/home/imagenes/CARS_DATABASES/KITTI_DATABASE/';
OUTPUT_DATA_PATH =  '../KITTI_CARS_DETECTION_EXPERIMENTS';
PREPARED_DATA_PATH =  'KITTI_TRAINING_DATA';



SHRINKAGE = 0.05;
FRAC_FEATURES = 1/32;
D = 8;
T = 1024;
N = 7500;
NA = 30000;
costsAlpha = 1;
costsBeta = 3;
costsGamma = 3;

TRAINED_DETECTOR_FILE = fullfile(OUTPUT_DATA_PATH, ...
                                 sprintf('BADACOST_%d_%d_%d_D_%d_T_%d_N_%d_NA_%d_S_%4.4f_F_%4.4f', costsAlpha, costsBeta, costsGamma, D, T, N, NA, SHRINKAGE, FRAC_FEATURES), ...
                                 'KITTI_SHRINKAGE_0.050000_RESAMPLING_1.000000_ASPECT_RATIO_1.750000_Detector.mat');
det = load(TRAINED_DETECTOR_FILE);


% Test for real image: 
img_name = 'coche_solo1.png';
I = imread(img_name);
det.detector.opts.pPyramid.nOctUp = 0;
P=chnsPyramid(I, det.detector.opts.pPyramid);
pad = det.detector.opts.pPyramid.pad;
shrink = det.detector.opts.pPyramid.pChns.shrink;
save_acf_channels(P.data{1}, pad, shrink, 'acfChannelsScale0_coche_solo1_png.yaml');


P.data{1}