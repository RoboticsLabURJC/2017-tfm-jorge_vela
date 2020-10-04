function[ ] = mat2yml(variable, fileName)
    matlab2opencv(variable.nPerOct, fileName, "w", "nPerOct");
    matlab2opencv(variable.nOctUp, fileName, "a", "nOctUp");
    matlab2opencv(variable.nApprox, fileName, "a", "nApprox");
    matlab2opencv(variable.lambdas, fileName, "a", "lambdas");
    matlab2opencv(variable.pad, fileName, "a", "pad");
    matlab2opencv(variable.minDs, fileName, "a", "minDs");
    matlab2opencv(variable.smooth, fileName, "a", "smooth");
    matlab2opencv(variable.concat, fileName, "a", "concat");
    matlab2opencv(variable.complete, fileName, "a", "complete");  
    
    file = fopen( fileName, 'a');
    fprintf( file, '    %s: \n', "pChns.pColor");
    fprintf( file, '        enabled: %d\n', variable.pChns.pColor.enabled);
    fprintf( file, '        smooth: %d\n', variable.pChns.pColor.smooth);
    fprintf( file, '        colorSpace: %s\n', variable.pChns.pColor.colorSpace);
    fclose(file);
    
    file = fopen( fileName, 'a');
    fprintf( file, '    %s: \n', "pChns.pGradMag");
    fprintf( file, '        enabled: %d\n', variable.pChns.pGradMag.enabled);
    fprintf( file, '        colorChn: %d\n', variable.pChns.pGradMag.colorChn);
    fprintf( file, '        normRad: %d\n', variable.pChns.pGradMag.normRad);
    fprintf( file, '        normConst: %d\n', variable.pChns.pGradMag.normConst);
    fprintf( file, '        full: %d\n', variable.pChns.pGradMag.full);
    fclose(file);   

    
    file = fopen( fileName, 'a');
    fprintf( file, '    %s: \n', "pChns.pGradHist");
    fprintf( file, '        enabled: %d\n', variable.pChns.pGradHist.enabled);
    fprintf( file, '        nOrients: %d\n', variable.pChns.pGradHist.nOrients);
    fprintf( file, '        softBin: %d\n', variable.pChns.pGradHist.softBin);
    fprintf( file, '        useHog: %d\n', variable.pChns.pGradHist.useHog);
    fprintf( file, '        clipHog: %d\n', variable.pChns.pGradHist.clipHog);
    fclose(file);   
    
    file = fopen( fileName, 'a');
    fprintf( file, '    %s: \n', "pChns.shrink");
    fprintf( file, '        data: %d\n', variable.pChns.shrink);
    fclose(file); 

    file = fopen( fileName, 'a');
    fprintf( file, '    %s: \n', "pChns.complete");
    fprintf( file, '        data: %d\n', variable.pChns.complete);
    fclose(file); 
end



