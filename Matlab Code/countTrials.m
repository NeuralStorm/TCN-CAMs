load("subj_trial_meta.mat")
fields = fieldnames(TrialData);
trials = zeros(20,1);
for i = 1:20
    sub = TrialData.(fields{i});
    trials(i) = size(sub.response_times,1);
end