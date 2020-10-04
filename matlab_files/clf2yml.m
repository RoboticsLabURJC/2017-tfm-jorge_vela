function[ ] = clf2yml(variable, fileName)
    matlab2opencv(variable.fids, fileName, "w", "fids");
    matlab2opencv(variable.thrs, fileName, "a", "thrs");
    matlab2opencv(variable.child, fileName, "a", "child");
    matlab2opencv(variable.hs, fileName, "a", "hs");
    matlab2opencv(variable.weights, fileName, "a", "weights");
    matlab2opencv(variable.depth, fileName, "a", "depth");
    matlab2opencv(variable.treeDepth, fileName, "a", "treeDepth");
    matlab2opencv(variable.num_classes, fileName, "a", "num_classes");
    matlab2opencv(variable.Cprime, fileName, "a", "Cprime");  
    matlab2opencv(variable.Y, fileName, "a", "Y");  
    matlab2opencv(variable.wl_weights, fileName, "a", "w1_weights");  
    matlab2opencv(variable.weak_learner_type, fileName, "a", "weak_learner_type");  
    matlab2opencv(variable.aRatio, fileName, "a", "aRatio");  
    matlab2opencv(variable.aRatioFixedWidth, fileName, "a", "aRatioFixedWidth");  

end
