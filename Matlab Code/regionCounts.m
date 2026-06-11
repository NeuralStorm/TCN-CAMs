allCounts = zeros(15,50);
ROIs = strings(1,50);
currMax = 1;
subjs = 1:20;
for subT = 1:length(subjs)
    sub = subjs(subT);
    if sub < 10
        load("data/GTH_s0" + sub + "_decision_power_struct_nobs.mat");
        fieldname = "s0" + sub;
    else
        load("data/GTH_s" + sub + "_decision_power_struct_nobs.mat");
        fieldname = "s" + sub;
    end

    regions = power_struct.anat.ROIs;
    
    [u, ~, idx] = unique(regions);
    counts = accumarray(idx, 1);

    for i = 1:size(counts, 1)
        a = u(i);
        a = string(a{1});
        if ismember(a, ROIs)
            allCounts(subT, find(ROIs == a)) = counts(i);
        else
            ROIs(currMax) = a;
            currMax = currMax + 1;
            allCounts(subT, find(ROIs == a)) = counts(i);
        end
    end
end