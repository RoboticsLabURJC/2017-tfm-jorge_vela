function save_acf_channels(acf_channels, filename)

sz = size(acf_channels);
for i=1:sz(3) 
   flag = 'a';
   if (i == 1)
       flag = 'w'
   end
   eval(['acf_channel_' num2str(i) '= acf_channels(:,:,i)']);
   eval(['matlab2opencv( ' 'acf_channel_'  num2str(i), ',filename, flag)']); 
end