load("subj_trial_meta.mat")
times = [];
vars = [];
fields = fieldnames(TrialData);
subjs = 1:20;
for subT = 1:length(subjs)
    sub = subjs(subT);
    
    response_time = mean(TrialData.(fields{subT}).response_times);
    response_var = std(TrialData.(fields{subT}).response_times);
    times = [times; response_time];
    vars = [vars; response_var];

    
end

