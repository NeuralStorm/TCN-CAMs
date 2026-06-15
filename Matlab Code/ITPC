% test to see whether decision (measured by proxy as button press) is
% always made at the same phase of the delta cycle.
clear all
% close all
clc

submean = [];
peakchans = {};
per_reg = {};

num_bootstrap = 1000;
desired_freqs = 1:4;

peak_deltas = [3     2     3     3     4     4     4     2     2     3     4     2     3     3     3     4];


for j  = 13:14
    loadedData = load(['dEEG00', int2str(j),  '/subject_00', int2str(j), '_data.mat']);
    if j < 5
        behFile = dir("dEEG00" + int2str(j) + "/*/behavior.xlsx");
    elseif j < 10
        behFile = dir("dEEG00" + int2str(j) + "/**/*00" + int2str(j) + "_*_.csv");
    else
        behFile = dir("dEEG00" + int2str(j) + "/**/*0" + int2str(j) + "_*_.csv");
    end
    tic

    SUBID = "s"+ num2str(j);
    
    behTable = readtable(append(behFile.folder, "\", behFile.name));
    if j > 4
        timeout = behTable(1:200,11);
        trial_outcome = behTable(1:200,12);
        loadedData.flash.data = loadedData.flash.acq.data;
        loadedData.flash.time = loadedData.flash.acq.time;
    
        trial_outcome = trial_outcome{:,:};
        timeout = timeout{:,:};
    else
        timeout = behTable(1:200,12);
        trial_outcome = behTable(1:200,13);
        trial_outcome = table2array(trial_outcome);
        timeout = table2array(timeout);
    end
    
    s = sort(loadedData.flash.data);
    s = s(floor(length(s) * .8));
    idx = loadedData.flash.data>s;
    loadedData.flash.data(idx) = s;
    m = loadedData.flash.data;
    
    edges = (1:100) * .01 * (max(m) - min(m)) + min(m);
    idx = discretize(m, edges);
    counts = histcounts(idx, 1:100); 
    
    [min_count, min_idx] = min(counts);
    flash_thresh = min_idx * .01 * (max(m) - min(m)) + min(m);
    flash_thresh
    trial_outcome = trial_outcome(~isnan(trial_outcome));
    trial_outcome(trial_outcome == 10, 1) = -1;
    trial_outcome(trial_outcome == -1, 1) = 1;
    trial_outcome(trial_outcome ~= 1, 1) = 0;
    trial_outcome(find(timeout == "Timeout"), 1) = -1;
    
    
    
    crossing_indices = find(loadedData.flash.data(1:end-1) < flash_thresh & loadedData.flash.data(2:end) >= flash_thresh | loadedData.flash.data(1:end-1) >= flash_thresh & loadedData.flash.data(2:end) < flash_thresh);
    
    differences = abs(diff(crossing_indices));
    
    indices = find(differences >= 2000);
    values = [crossing_indices(1), crossing_indices(indices(1)), crossing_indices(indices + 1)];
    size(values)
    values = values(:,size(values,2) - 399:end);
    trial_start = values(2:2:end); %change first value to 1 for stim onset
    times = loadedData.flash.time(trial_start);
    
    eeg_start_indices = cell2mat(arrayfun(@(x) convertTimes(x, loadedData.eeg.time), times, "un", 0));
    eeg_start_indices = eeg_start_indices(~isnan(eeg_start_indices));
    
    size(eeg_start_indices)
    data = arrayfun(@(x) trialData(x, loadedData.eeg.data, loadedData.eeg.lslInfo.nominal_srate), eeg_start_indices, "un", 0);
    "data shape"
    size(data)
    
    recordingFrequency = str2num(loadedData.eeg.lslInfo.nominal_srate);
    num_trials = size(data,2);
    Ts = 1/recordingFrequency;
    
    parsedData = [];
    for i = 1:num_trials
        %i
        x = data(i);
        x = cell2mat(x);
        parsedData = [parsedData; reshape(x, [1, size(x,1), size(x,2)])];
    end
    fs = recordingFrequency;
    [nTrials, nChannels, nTime] = size(parsedData);
    data = [];
    data.label = arrayfun(@(x) sprintf('chan%d', x), 1:nChannels, 'UniformOutput', false);
    data.fsample = fs;
    
    for tr = 1:nTrials
        data.trial{tr} = squeeze(parsedData(tr, :, :));   % [channels x time]
        data.time{tr}  = (0:nTime-1) / fs;         % adjust if you have real time axis
    end


    %% how consistent is delta phase within channel over trials? Inter-trial coherence
    keep_trials = find(~isnan(behTable.DecisionOnset(1:end -4)));
    cfg = [];
    cfg.trials = keep_trials;
    cfg.method = 'wavelet'; % wavelet based approach
    cfg.toi    = data.time{1};
    cfg.output = 'fourier';
    cfg.pad = 'nextpow2';
    % cfg.foi = 1:1:4;
    cfg.foi = desired_freqs;
    freq       = ft_freqanalysis(cfg, data);

    % make a new FieldTrip-style data structure containing the ITC
    % copy the descriptive fields over from the frequency decomposition
    itc           = [];
    itc.label     = freq.label;
    itc.freq      = freq.freq;
    itc.time      = freq.time;
    itc.dimord    = 'chan_freq_time';


    F = freq.fourierspctrm;   % copy the Fourier spectrum
    N = size(F,1);           % number of trials
    
    % compute inter-trial phase coherence (itpc)
    itc.itpc      = F./abs(F);         % divide by amplitude
    itc.itpc      = sum(itc.itpc,1);   % sum angles
    itc.itpc      = abs(itc.itpc)/N;   % take the absolute value and normalize
    itc.itpc      = squeeze(itc.itpc); % remove the first singleton dimension
   
    %% keep data output: mean ITPC per region at time 0 at subject's peak delta
    
    peak_delta_sub =  peak_deltas(j-5);    
    choice_idx = find(itc.time == 4);
    % individual values for peak delta
    channel_values_choice = itc.itpc(:,peak_delta_sub,choice_idx);

    %% make a control distribution for this subject
    channel_values_choice_shuffled = [];
    for control_dist = 1:num_bootstrap
        % shuffle the trials by a random #
        % generate random distribution
        data_shuffled = data;
        shift_by = randi(size(data.trial{1,1},2),numel(data.trial),1);
        for trial_idx = 1:numel(data.trial)
            data_shuffled.trial{1,trial_idx} = circshift(data.trial{1,trial_idx},shift_by(trial_idx,1),2);
        end
        % calculate ITPC across these shuffled trials
        cfg        = [];
        cfg.trials = keep_trials;
        cfg.method = 'wavelet'; % wavelet based approach
        cfg.toi    = data.time{1}; % times on which the analysis should be centered, 10 ms sliding here
        cfg.output = 'fourier';
        cfg.pad = 'nextpow2';
        % cfg.foi =  1:1:4;
        cfg.foi = desired_freqs;
        freq       = ft_freqanalysis(cfg, data_shuffled);

        % make a new FieldTrip-style data structure containing the ITC
        % copy the descriptive fields over from the frequency decomposition
        itc_shuffled           = [];
        itc_shuffled.label     = freq.label;
        itc_shuffled.freq      = freq.freq;
        itc_shuffled.time      = freq.time;
        itc_shuffled.dimord    = 'chan_freq_time';


        F = freq.fourierspctrm;   % copy the Fourier spectrum
        N = size(F,1);           % number of trials
    
        % compute inter-trial phase coherence (itpc)
        itc_shuffled.itpc      = F./abs(F);         % divide by amplitude
        itc_shuffled.itpc      = sum(itc_shuffled.itpc,1);   % sum angles
        itc_shuffled.itpc      = abs(itc_shuffled.itpc)/N;   % take the absolute value and normalize
        itc_shuffled.itpc      = squeeze(itc_shuffled.itpc); % remove the first singleton dimension
        
        channel_values_choice_shuffled(:,control_dist) = itc_shuffled.itpc(:,peak_delta_sub,choice_idx);
    end
    itpc_channel_output.choice.(SUBID).true_vals = channel_values_choice;
    itpc_channel_output.choice.(SUBID).controls = channel_values_choice_shuffled;
    % figure out channel significance
    threshold = prctile(channel_values_choice_shuffled,95,2);
    itpc_channel_output.choice.(SUBID).choice_significance = [];
    for chan_idx = 1:numel(channel_values_choice)
        itpc_channel_output.choice.(SUBID).choice_significance(chan_idx,1) = channel_values_choice(chan_idx) >= threshold(chan_idx);
    end
    
    
    %% REPEAT, ZEROING ON CHOICE
    % shift each trial by reaction time
    fix_times = behTable.DecisionOnset(1:end - 4) - behTable.TrialOnset(1:end - 4);
    cfg = [];
    cfg.offset = fix_times;
    fix_data = ft_redefinetrial(cfg,data);

    cfg        = [];
    cfg.trials = keep_trials;
    cfg.method = 'wavelet'; % wavelet based approach
    cfg.toi    = fix_data.time{2}; % times on which the analysis should be centered, 10 ms sliding here
    cfg.output = 'fourier';
    % cfg.foi = 1:1:4;
    cfg.foi = desired_freqs;
    cfg.pad = 'nextpow2';
    freq_fix = ft_freqanalysis(cfg, fix_data);

    % make a new FieldTrip-style data structure containing the ITC
    % copy the descriptive fields over from the frequency decomposition
    
    itc_fix           = [];
    itc_fix.label     = freq_fix.label;
    itc_fix.freq      = freq_fix.freq;
    itc_fix.time      = freq_fix.time;
    itc_fix.dimord    = 'chan_freq_time';


    F = freq_fix.fourierspctrm;   % copy the Fourier spectrum
    N = size(F,1);           % number of trials
    
    % compute inter-trial phase coherence (itpc)
    itc_fix.itpc      = F./abs(F);         % divide by amplitude
    itc_fix.itpc      = sum(itc_fix.itpc,1);   % sum angles
    itc_fix.itpc      = abs(itc_fix.itpc)/N;   % take the absolute value and normalize
    itc_fix.itpc      = squeeze(itc_fix.itpc); % remove the first singleton dimension
   
    %% keep data output: mean ITPC per region at time 0 at subject's peak delta
    choice_idx = find(itc_fix.time == 4);
    % individual values for peak delta
    channel_values_fix = itc_fix.itpc(:,peak_delta_sub,choice_idx);

    %% make a control distribution for this subject
    channel_values_fix_shuffled = [];
    for control_dist = 1:num_bootstrap
        % shuffle the trials by a random #
        % generate random distribution
        data_shuffled_fix = fix_data;
        shift_by = randi(size(fix_data.trial{1,1},2),numel(fix_data.trial),1);
        for trial_idx = 1:numel(fix_data.trial)
            data_shuffled_fix.trial{1,trial_idx} = circshift(fix_data.trial{1,trial_idx},shift_by(trial_idx,1),2);
        end
        % calculate ITPC across these shuffled trials
        cfg        = [];
        cfg.trials = keep_trials;
        cfg.method = 'wavelet'; % wavelet based approach
        cfg.toi    =fix_data.time{2}; % times on which the analysis should be centered, 10 ms sliding here
        cfg.output = 'fourier';
        % cfg.foi =  1:1:4;
        cfg.foi = desired_freqs;
        cfg.pad = 'nextpow2';
        freq_fix       = ft_freqanalysis(cfg, data_shuffled_fix);

        % make a new FieldTrip-style data structure containing the ITC
        % copy the descriptive fields over from the frequency decomposition
        itc_shuffled_fix           = [];
        itc_shuffled_fix.label     = freq_fix.label;
        itc_shuffled_fix.freq      = freq_fix.freq;
        itc_shuffled_fix.time      = freq_fix.time;
        itc_shuffled_fix.dimord    = 'chan_freq_time';


        F = freq_fix.fourierspctrm;   % copy the Fourier spectrum
        N = size(F,1);           % number of trials
    
        % compute inter-trial phase coherence (itpc)
        itc_shuffled_fix.itpc      = F./abs(F);         % divide by amplitude
        itc_shuffled_fix.itpc      = sum(itc_shuffled_fix.itpc,1);   % sum angles
        itc_shuffled_fix.itpc      = abs(itc_shuffled_fix.itpc)/N;   % take the absolute value and normalize
        itc_shuffled_fix.itpc      = squeeze(itc_shuffled_fix.itpc); % remove the first singleton dimension 
               
        channel_values_fix_shuffled(:,control_dist) = itc_shuffled_fix.itpc(:,peak_delta_sub,choice_idx);
    end
    itpc_channel_output.fix.(SUBID).true_vals = channel_values_fix;
    itpc_channel_output.fix.(SUBID).controls = channel_values_fix_shuffled;
    % figure out channel significance
    threshold = prctile(channel_values_fix_shuffled,95,2);
    itpc_channel_output.fix.(SUBID).fix_significance = [];
    for chan_idx = 1:numel(channel_values_fix)
        itpc_channel_output.fix.(SUBID).fix_significance(chan_idx,1) = channel_values_fix(chan_idx) >= threshold(chan_idx);
    end
end
%% save output
itpc_channel_output.band_name = 'delta';
itpc_channel_output.freqs = desired_freqs;

save(['itpc_choice_vs_fixed_1314_', num2str(num_bootstrap), 'iters_'],'itpc_channel_output');



subjects= fieldnames(itpc_channel_output.choice);
for sub_idx = 1:numel(subjects)
    avg_magnitude_choice(sub_idx,1) = mean(itpc_channel_output.choice.(subjects{sub_idx}).true_vals,"omitmissing");
    avg_magnitude_fix(sub_idx,1) = mean(itpc_channel_output.fix.(subjects{sub_idx}).true_vals,"omitmissing");

    avg_magnitude_choice_control(sub_idx,1) = mean(itpc_channel_output.choice.(subjects{sub_idx}).controls,[1,2],"omitmissing");
    avg_magnitude_fix_control(sub_idx,1) = mean(itpc_channel_output.fix.(subjects{sub_idx}).controls,[1,2],"omitmissing");

    per_locked_choice(sub_idx,1) = sum(itpc_channel_output.choice.(subjects{sub_idx}).choice_significance,"omitmissing")/numel(itpc_channel_output.choice.(subjects{sub_idx}).true_vals)*100;
    per_locked_fixed(sub_idx,1) = sum(itpc_channel_output.fix.(subjects{sub_idx}).fix_significance,"omitmissing")/numel(itpc_channel_output.fix.(subjects{sub_idx}).true_vals)*100;
end



function times = convertTimes(z, y)
    times = min(find(abs(z - y) < .1));
    if isempty(times)
        times = NaN;
        "couldn't find time"
    end 
end


function trialed = trialData(start, data, recordingFrequency)
    trialed = data(:,start - (5 * str2num(recordingFrequency)):start + (3 * str2num(recordingFrequency)));% - str2num(recordingFrequency));
end
