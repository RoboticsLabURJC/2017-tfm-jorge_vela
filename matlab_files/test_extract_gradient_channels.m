
TOOLBOX_BADACOST_PATH = '/home/jmbuena/matlab/toolbox.badacost.public';
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


det.detector.opts.pPyramid.nOctUp = 0;

% Test for real image: 
img_name = 'index.jpeg';
I = imread(img_name);
Igray = rgb2gray(I);

%--------------------------
% ---- Test color 1
%--------------------------

% compute gradient magnitude channel
p_gm = det.detector.opts.pPyramid.pChns.pGradMag; 
full=0;
if (isfield(p_gm,'full'))
  full=p.full; 
end
[M,O] = gradientMag(single(I), p_gm.colorChn, p_gm.normRad, p_gm.normConst, full);

% compute gradient histgoram channels
p_gh = det.detector.opts.pPyramid.pChns.pGradHist; 
binSize = p_gh.binSize; 
if (isempty(binSize))
  binSize = det.detector.opts.pPyramid.pChns.shrink; 
end
H = gradientHist(M, O, binSize, p_gh.nOrients, p_gh.softBin, p_gh.useHog, p_gh.clipHog, full);
normRad = p_gm.normRad;
normConst = p_gm.normConst;
save_gradient_channels(M, O, H, normRad, normConst, 'index_jpeg_GradientChannels.yaml');

%--------------------------
% ---- Test gray
%--------------------------

% Test for real image: 

% compute gradient magnitude channel
[M,O] = gradientMag(single(Igray), p_gm.colorChn, p_gm.normRad, p_gm.normConst, full);

% compute gradient histgoram channels
H = gradientHist(M, O, binSize, p_gh.nOrients, p_gh.softBin, p_gh.useHog, p_gh.clipHog, full);
normRad = p_gm.normRad;
normConst = p_gm.normConst;
save_gradient_channels(M, O, H, normRad, normConst, 'index_jpeg_gray_GradientChannels.yaml');

%--------------------------
% ---- Test color
%--------------------------

% Test for real image: 

% compute gradient magnitude channel
p_gm.normRad = 0;
[M,O] = gradientMag(single(I), p_gm.colorChn, p_gm.normRad, p_gm.normConst, full);

% compute gradient histgoram channels
H = gradientHist(M, O, binSize, p_gh.nOrients, p_gh.softBin, p_gh.useHog, p_gh.clipHog, full);
normRad = p_gm.normRad;
normConst = p_gm.normConst;
save_gradient_channels(M, O, H, normRad, normConst, 'index_jpeg_gray_GradientChannels_normRad_0.yaml');

%--------------------------
% ---- Test color
%--------------------------

% Test for real image: 

% compute gradient magnitude channel
p_gm.normRad = 5;
p_gm.normConst = 0.07;
[M,O] = gradientMag(single(I), p_gm.colorChn, p_gm.normRad, p_gm.normConst, full);

% compute gradient histgoram channels
H = gradientHist(M, O, binSize, p_gh.nOrients, p_gh.softBin, p_gh.useHog, p_gh.clipHog, full);
normRad = p_gm.normRad;
normConst = p_gm.normConst;
save_gradient_channels(M, O, H, normRad, normConst, 'index_jpeg_gray_GradientChannels_normConst_0_07.yaml');

    