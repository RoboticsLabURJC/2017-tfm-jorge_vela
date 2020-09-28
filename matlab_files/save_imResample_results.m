function save_imResample_results(Iresampled, scale, method, norm, filename)

matlab2opencv(scale, filename, 'w');
matlab2opencv(method, filename, 'a');
matlab2opencv(norm, filename, 'a');

sz = size(Iresampled);
for i=1:sz(3) 
   eval(['img_resampled_' num2str(i) '= Iresampled(:,:,i);']);
   eval(['matlab2opencv( ' 'img_resampled_'  num2str(i), ',filename, ''a'')']); 
end