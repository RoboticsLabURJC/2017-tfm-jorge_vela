function[ ] = clf2yml(variable, fileName)
    matlab2opencv(variable.clf.fids, fileName, "w", "fids");
    matlab2opencv(variable.clf.thrs, fileName, "a", "thrs");
    matlab2opencv(variable.clf.child, fileName, "a", "child");
    matlab2opencv(variable.clf.hs, fileName, "a", "hs");
    matlab2opencv(variable.clf.weights, fileName, "a", "weights");
    matlab2opencv(variable.clf.depth, fileName, "a", "depth");
    matlab2opencv(variable.clf.treeDepth, fileName, "a", "treeDepth");
    matlab2opencv(variable.clf.num_classes, fileName, "a", "num_classes");
    matlab2opencv(variable.clf.Cprime, fileName, "a", "Cprime");  
    matlab2opencv(variable.clf.Y, fileName, "a", "Y");  
    matlab2opencv(variable.clf.wl_weights, fileName, "a", "w1_weights");  
    matlab2opencv(variable.clf.weak_learner_type, fileName, "a", "weak_learner_type");  
    matlab2opencv(variable.clf.aRatio, fileName, "a", "aRatio");  
    matlab2opencv(variable.clf.aRatioFixedWidth, fileName, "a", "aRatioFixedWidth");  

    matlab2opencv(variable.opts.pPyramid.nPerOct, fileName, "a", "nPerOct");
    matlab2opencv(variable.opts.pPyramid.nOctUp, fileName, "a", "nOctUp");
    matlab2opencv(variable.opts.pPyramid.nApprox, fileName, "a", "nApprox");
    matlab2opencv(variable.opts.pPyramid.lambdas, fileName, "a", "lambdas");
    matlab2opencv(variable.opts.pPyramid.pad, fileName, "a", "pad");
    matlab2opencv(variable.opts.pPyramid.minDs, fileName, "a", "minDs");
    matlab2opencv(variable.opts.pPyramid.smooth, fileName, "a", "smooth");
    matlab2opencv(variable.opts.pPyramid.concat, fileName, "a", "concat");
    matlab2opencv(variable.opts.pPyramid.complete, fileName, "a", "complete");      
    
    
    file = fopen( fileName, 'a');
    fprintf( file, '    %s: \n', "pChns.pColor");
    fprintf( file, '        enabled: %d\n', variable.opts.pPyramid.pChns.pColor.enabled);
    fprintf( file, '        smooth: %d\n', variable.opts.pPyramid.pChns.pColor.smooth);
    fprintf( file, '        colorSpace: %s\n', variable.opts.pPyramid.pChns.pColor.colorSpace);
    fclose(file);
    
    file = fopen( fileName, 'a');
    fprintf( file, '    %s: \n', "pChns.pGradMag");
    fprintf( file, '        enabled: %d\n', variable.opts.pPyramid.pChns.pGradMag.enabled);
    fprintf( file, '        colorChn: %d\n', variable.opts.pPyramid.pChns.pGradMag.colorChn);
    fprintf( file, '        normRad: %d\n', variable.opts.pPyramid.pChns.pGradMag.normRad);
    fprintf( file, '        normConst: %d\n', variable.opts.pPyramid.pChns.pGradMag.normConst);
    fprintf( file, '        full: %d\n', variable.opts.pPyramid.pChns.pGradMag.full);
    fclose(file);   

    
    file = fopen( fileName, 'a');
    fprintf( file, '    %s: \n', "pChns.pGradHist");
    fprintf( file, '        enabled: %d\n', variable.opts.pPyramid.pChns.pGradHist.enabled);
    fprintf( file, '        nOrients: %d\n', variable.opts.pPyramid.pChns.pGradHist.nOrients);
    fprintf( file, '        softBin: %d\n', variable.opts.pPyramid.pChns.pGradHist.softBin);
    fprintf( file, '        useHog: %d\n', variable.opts.pPyramid.pChns.pGradHist.useHog);
    fprintf( file, '        clipHog: %d\n', variable.opts.pPyramid.pChns.pGradHist.clipHog);
    fclose(file);   
    
    file = fopen( fileName, 'a');
    fprintf( file, '    %s: \n', "pChns.shrink");
    fprintf( file, '        data: %d\n', variable.opts.pPyramid.pChns.shrink);
    fclose(file); 

    file = fopen( fileName, 'a');
    fprintf( file, '    %s: \n', "pChns.complete");
    fprintf( file, '        data: %d\n', variable.opts.pPyramid.pChns.complete);
    fclose(file);     
    
end






















