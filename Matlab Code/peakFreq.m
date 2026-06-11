peaks = []
for j = 13:14
    fftLength = 3;
    int2str(j)
    loadedData = load(['dEEG00', int2str(j),  '/subject_00', int2str(j), '_data.mat']);
    
    if j < 5
        behFile = dir("dEEG00" + int2str(j) + "/*/behavior.xlsx");
    elseif j < 10
        behFile = dir("dEEG00" + int2str(j) + "/**/*00" + int2str(j) + "_*_.csv");
    else
        behFile = dir("dEEG00" + int2str(j) + "/**/*0" + int2str(j) + "_*_.csv");
    end
    trial_outcome = readtable(append(behFile.folder, "\", behFile.name));
    if j > 4
        timeout = trial_outcome(1:200,11);
        trial_outcome = trial_outcome(1:200,12);
        loadedData.flash.data = loadedData.flash.acq.data;
        loadedData.flash.time = loadedData.flash.acq.time;
    
        trial_outcome = trial_outcome{:,:};
        timeout = timeout{:,:};
    else
        timeout = trial_outcome(1:200,12);
        trial_outcome = trial_outcome(1:200,13);
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
    
    
    recordingFrequency = str2num(loadedData.eeg.lslInfo.nominal_srate);
    num_trials = size(data,2);
    Ts = 1/recordingFrequency;
    t = 0:Ts:1;
    numElectrodes = 64;
    all_trials_phase_control_safe = zeros(num_trials, numElectrodes,10,recordingFrequency * fftLength + 1);
    all_trials_phase_safe = zeros(num_trials, numElectrodes,recordingFrequency * fftLength + 1);
    
    all_trials_phase_control_gamble = zeros(num_trials, numElectrodes,10, recordingFrequency * fftLength + 1);
    all_trials_phase_gamble = zeros(num_trials, numElectrodes,recordingFrequency * fftLength + 1);
    
    all_trials_phase_control = zeros(num_trials, numElectrodes,10,recordingFrequency * fftLength + 1);
    all_trials_phase = zeros(num_trials, numElectrodes,recordingFrequency * fftLength + 1);
    timeouts = 0;
    
    for i = 1:num_trials
        %i
        x = data(i);
        x = cell2mat(x);
        all_electrodes = zeros(size(x,1), recordingFrequency * fftLength + 1);
        all_phase_electrodes = zeros(size(x,1), recordingFrequency * fftLength + 1);
        all_phase_electrodes_control = zeros(size(x,1),10, recordingFrequency * fftLength + 1);
        for k = 1:size(x,1)
            trial = x(k,:);
            trial = squeeze(trial);
            trial = fft(trial);
            trial = squeeze(trial);
            if size(trial,2) < recordingFrequency * fftLength + 1
                all_phase_electrodes(k,:) = padarray(trial, [0,recordingFrequency * fftLength + 1 - size(trial,2)],0, "post");
                continue
            end
            all_phase_electrodes(k,:) = trial;
        end
    
        
        if trial_outcome(i) == 1
            all_trials_phase_gamble(i,:,:) = all_phase_electrodes;
            all_trials_phase(i,:,:) = all_phase_electrodes;
            % all_trials_phase_control_gamble(i,:,:) = mean(all_phase_electrodes_control);
        elseif trial_outcome(i) == 0
            all_trials_phase_safe(i,:,:) = all_phase_electrodes;
            all_trials_phase(i,:,:) = all_phase_electrodes;
            % all_trials_phase_control_safe(i,:,:) = mean(all_phase_electrodes_control);
        else
            timeouts = timeouts + 1;
        end
    end
    timeouts
    eegData = loadedData.eeg.data;
    % save("StimOffset/" + int2str(j) + ","  + int2str(fftLength) + ", control " + int2str(control) + ", post ", "all_trials_phase", "trial_outcome", "eegData", "eeg_start_indices")
    
    maxFreq = 4;
    minFreq = .5;
    freqBinSize = 1/fftLength;
    averageTrialsPhase = mean(all_trials_phase,2);
    hz=  (0:length(x)-1) * (recordingFrequency/length(trial));
    modes = [];
    for i = 2:4
        modes = [modes mode(abs(averageTrialsPhase((hz > (i - .5)) & (hz < (i + 0.5)))))];
    end
    [max_val, idx] = max(modes);
    peaks = [peaks, idx];
end


function times = convertTimes(z, y)
    times = min(find(abs(z - y) < .1));
    if isempty(times)
        times = NaN;
        "couldn't find time"
    end 
end

function trialed = trialData(start, data, recordingFrequency)
    trialed = data(:,start - (3 * str2num(recordingFrequency)):start);% - str2num(recordingFrequency));
end
