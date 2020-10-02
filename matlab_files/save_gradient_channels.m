function save_gradient_channels(M, O, H, normRad, normConst, filename)

matlab2opencv(M, filename, 'w');
matlab2opencv(O, filename, 'a');
matlab2opencv(normRad, filename, 'a');
matlab2opencv(normConst, filename, 'a');

sz = size(H);
for i=1:sz(3) 
   eval(['H_' num2str(i) '= H(:,:,i);']);
   eval(['matlab2opencv( ' 'H_'  num2str(i), ',filename, ''a'');']); 
end