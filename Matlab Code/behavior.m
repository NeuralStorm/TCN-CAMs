f = figure;
T = tiledlayout(4, 4,"TileSpacing","loose", "Padding", "loose");
k = 0;
postVonMises = zeros(16,1);
preVonMises = zeros(16,1);
timeoutsDiff = {};
nonTimeoutsDiff = {};
kuiper = zeros(16,1);
ktest =  zeros(16,1);
reactionTimes = zeros(16,1);
deliberationTimes =  zeros(16,1);
relativeRewards = zeros(16,1);
mistakesTotal = zeros(16,1);
timeoutsTotal = zeros(16,1);
phaseMeans = zeros(16,1);
allPhases = zeros(16, 200);
phaseDev = zeros(16,1);

timeoutDuration = zeros(16,1);
nonDuration = zeros(16,1);
for j = 6:21
    fftLength = 4;
    
    k = k + 1;
    loadedData = load(['dEEG00', int2str(j),  '/subject_00', int2str(j), '_data.mat']);
    post = load("StartOffset/" + int2str(j) + ","  + int2str(fftLength) + ", control " + int2str(false) + ", post " + int2str(true), "all_trials_phase", "trial_outcome", "eegData", "eeg_start_indices");
    stimOnset = load("StimOffset/" + int2str(j) + ","  + int2str(fftLength) + ", control " + int2str(false) + ", post " + int2str(true), "all_trials_phase", "trial_outcome", "eegData", "eeg_start_indices");
    



    if j < 5
        behFile = dir("dEEG00" + int2str(j) + "/*/behavior.xlsx");
    elseif j < 10
        behFile = dir("dEEG00" + int2str(j) + "/**/*00" + int2str(j) + "_*_.csv");
    else
        behFile = dir("dEEG00" + int2str(j) + "/**/*0" + int2str(j) + "_*_.csv");
    end

    
    opts = detectImportOptions(append(behFile.folder, "\", behFile.name));
    opts = setvartype(opts,'char');
    trial_outcome = readtable(append(behFile.folder, "\", behFile.name),opts);
    reactionTimes(j - 5) = sscanf(trial_outcome{end, "Round"}{1},"%*[^:]: %f");
    deliberationTimes(j - 5) = sscanf(trial_outcome{end - 1, "Round"}{1},"%*[^:]: %f");
    
    continue
    trial_outcome = readtable(append(behFile.folder, "\", behFile.name));
    
    relativeReward = zeros(200,1);
    for k = 1:200
        if strcmp(trial_outcome{k,"GambleSide"}{1}, trial_outcome{k,"Choice"}{1})
            relativeReward(k) = (trial_outcome{k,"GambleReward"} * (10 - trial_outcome{k,"FirstNumber"}) * .1) - 10;
        else
            relativeReward(k) = 10 - (trial_outcome{k,"GambleReward"} * (10 - trial_outcome{k,"FirstNumber"}) * .1);
        end
    end
    
    relativeRewards(j - 5) = mean(relativeReward);
    mistakesTotal(j - 5) = sum(relativeReward < 0);
    choice = trial_outcome{:,"Choice"};
    timeouts = choice == "Timeout";
    size(choice)
    timeoutsTotal(j - 5) = sum(timeouts);
    
    continue
    
    nexttile();
    amps = zeros(16,1);
    for i = .5:.25:4
        subj_peak_delta = i;
        cfg = [];
        cfg.bpfilter = 'yes';
        cfg.bpfreq = [subj_peak_delta-0.5, subj_peak_delta+0.5];
        cfg.bpfilttype = 'fir';
        %cfg.hilbert = 'angle';
        %cfg.trials = gamble_trials;
    
        data = [];
        data.trial = {post.eegData};
        data.time = {loadedData.eeg.time};
        data.label = string(1:64);
        data.fsample = 512;
        %data_delta = ft_preprocessing(cfg, data);
    
        % averaged delta
        %delta_average = data_delta;
        %delta_average.trial = mean(data_delta.avg);
        %delta_average.avg = mean(data_delta.avg);
        %delta_average.trial = cellfun(@(x) mean(x, 'omitmissing'), mat2cell(data_delta.avg), 'UniformOutput', false);
        %delta_average.label = {'average'};
        
        % now calculate phase
        cfg = [];
        cfg.bpfilter = 'yes';
        cfg.bpfilttype = 'fir';
        cfg.bpfreq = [subj_peak_delta-0.125, subj_peak_delta+0.125];
        
        cfg.hilbert = 'abs';
        %cfg.hilbert = 'angle';
        %cfg.trials = TrialData.(SUBID).correct_gamble;
        %data_phase = ft_preprocessing(cfg, data_delta);
        amp = ft_preprocessing(cfg, data);
       
        amps(round(i/.25 - 1)) = mean(amp.trial{1}(1,post.eeg_start_indices - str2num(loadedData.eeg.lslInfo.nominal_srate)));
    end 
    bar(amps);
    continue

    % change angle range from 0 to 2pi
    %plotResults(nanmean(data_phase.avg(:,post.eeg_start_indices - str2num(loadedData.eeg.lslInfo.nominal_srate)) + pi))
    %scatter(relativeReward, abs(sin(.5 * (data_phase.avg(1,post.eeg_start_indices - str2num(loadedData.eeg.lslInfo.nominal_srate)) + pi)) - sin(.5 * (mean(data_phase.avg(1,post.eeg_start_indices - str2num(loadedData.eeg.lslInfo.nominal_srate)) + pi)))))

    %scatter(relativeReward, data_phase.avg(1,post.eeg_start_indices - str2num(loadedData.eeg.lslInfo.nominal_srate)) + pi)
    %chain = data_phase.avg(1,post.eeg_start_indices - str2num(loadedData.eeg.lslInfo.nominal_srate));
    %scatter(relativeReward(find(post.trial_outcome)), chain(find(post.trial_outcome)) + pi)

    % if sum(timeouts) > 0
    %     timeouts = timeouts(1:200,:);
    %     scatter(zeros(sum(timeouts)), data_phase.avg(1,post.eeg_start_indices(timeouts) - str2num(loadedData.eeg.lslInfo.nominal_srate)) + pi)
    %     hold on
    %     scatter(ones(sum(~timeouts)), data_phase.avg(1,post.eeg_start_indices(~timeouts) - str2num(loadedData.eeg.lslInfo.nominal_srate)) + pi)
    %     nonTimeoutsDiff = [nonTimeoutsDiff, mean(abs(sin(.5 * (data_phase.avg(1,post.eeg_start_indices(~timeouts) - str2num(loadedData.eeg.lslInfo.nominal_srate)) + pi)) - sin(.5 * (mean(data_phase.avg(1,post.eeg_start_indices - str2num(loadedData.eeg.lslInfo.nominal_srate)) + pi)))))]
    % 
    %     timeoutsDiff = [timeoutsDiff, mean(abs(sin(.5 * (data_phase.avg(1,post.eeg_start_indices(timeouts) - str2num(loadedData.eeg.lslInfo.nominal_srate)) + pi)) - sin(.5 * (mean(data_phase.avg(1,post.eeg_start_indices - str2num(loadedData.eeg.lslInfo.nominal_srate)) + pi)))))];
    % 
    %     %bar([data_phase.avg(1,post.eeg_start_indices(timeouts) - str2num(loadedData.eeg.lslInfo.nominal_srate)) + pi; data_phase.avg(1,post.eeg_start_indices(~timeouts) - str2num(loadedData.eeg.lslInfo.nominal_srate)) + pi])
    % end
    timeouts = timeouts(1:200,:);
    print(sum(timeouts))
    % scatter(post.eeg_start_indices(~timeouts) - stimOnset.eeg_start_indices(~timeouts), abs(sin(.5 * (data_phase.avg(1,post.eeg_start_indices(~timeouts) - str2num(loadedData.eeg.lslInfo.nominal_srate)) + pi) - str2num(loadedData.eeg.lslInfo.nominal_srate)) + pi)) - sin(.5 * (mean(data_phase.avg(1,post.eeg_start_indices(~timeouts) - str2num(loadedData.eeg.lslInfo.nominal_srate)) + pi)))))
    
    % scatter(post.eeg_start_indices(~timeouts) - stimOnset.eeg_start_indices(~timeouts), relativeReward(~timeouts))
    phases = sin(.5 * (data_phase.avg(1,post.eeg_start_indices - str2num(loadedData.eeg.lslInfo.nominal_srate)) + pi));
    phaseMeans(j - 5) = mean(phases);
    phaseDev(j - 5) = std(phases);
    allPhases(j, :) = phases;


    choiceTime = trial_outcome{:,"ChoiceOnset"};
    decisionTime = trial_outcome{:,"FeedbackOnset"};
    choiceTime = str2double(choiceTime(1:end - 4));
    decisionTime = str2double(decisionTime(1:end - 4));

    timeToDecide = decisionTime - choiceTime;
    timeToDecide = timeToDecide(~isnan(timeToDecide));

    timeoutDuration(j) = mean(timeToDecide(timeouts)) - 1;
    nonDuration(j) = mean(timeToDecide(~timeouts)) - 1;
    %scatter(post.trial_outcome(~timeouts), abs(sin(.5 * (data_phase.avg(1,post.eeg_start_indices(~timeouts) - str2num(loadedData.eeg.lslInfo.nominal_srate)) + pi)) - sin(.5 * (mean(data_phase.avg(1,post.eeg_start_indices(~timeouts) - str2num(loadedData.eeg.lslInfo.nominal_srate)) + pi)))))
end

figure()

%significant if exclude outlier 3
scatter(reactionTimes, phaseDev)

%maybe if exclude 9 - look at individual trials
corr(relativeRewards, phaseDev)