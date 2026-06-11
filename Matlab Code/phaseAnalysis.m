function [] = phaseAnalysis(subject, fftLength, post, control)
    j = subject;
    j
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
    trial_start = values(1:2:end); %change first value to 1 for stim onset
    times = loadedData.flash.time(trial_start);
    
    if control
        times = min(times) + rand * (max(times) - min(times));
    end
    eeg_start_indices = cell2mat(arrayfun(@(x) convertTimes(x, loadedData.eeg.time), times, "un", 0));
    eeg_start_indices = eeg_start_indices(~isnan(eeg_start_indices));
    
    size(eeg_start_indices)
    data = arrayfun(@(x) trialData(x, loadedData.eeg.data, loadedData.eeg.lslInfo.nominal_srate, fftLength, post), eeg_start_indices, "un", 0);
    
    
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
            trial(abs(trial) < 1e-6) = 0;
            if size(trial,2) < recordingFrequency * fftLength + 1
                all_phase_electrodes(k,:) = padarray(trial, [0,recordingFrequency * fftLength + 1 - size(trial,2)],0, "post");
                continue
            end
            all_phase_electrodes(k,:) = trial;
        end
    
        
        if trial_outcome(i) == 1
            all_trials_phase_gamble(i,:,:) = all_phase_electrodes;
            % all_trials_phase_control_gamble(i,:,:) = mean(all_phase_electrodes_control);
        elseif trial_outcome(i) == 0
            all_trials_phase_safe(i,:,:) = all_phase_electrodes;
            % all_trials_phase_control_safe(i,:,:) = mean(all_phase_electrodes_control);
        else
            timeouts = timeouts + 1;
        end
        all_trials_phase(i,:,:) = all_phase_electrodes;
        % all_trials_phase_control(i,:,:) = mean(all_phase_electrodes_control, 1);
    end
    timeouts
    hold on
    eegData = loadedData.eeg.data;
    save("StimOffset/" + int2str(j) + ","  + int2str(fftLength) + ", control " + int2str(control) + ", post " + int2str(post), "all_trials_phase", "trial_outcome", "eegData", "eeg_start_indices")
    
    maxFreq = 4;
    minFreq = .5;
    freqBinSize = 1/fftLength;
    numPlots = (ceil(4/freqBinSize) + 1) - (floor(.5/freqBinSize) + 1) + 1;
    tiledlayout(ceil(numPlots/5),min(5, numPlots))
    size(all_trials_phase)
    for frequency = (ceil(.5/freqBinSize) + 1):(floor(4/freqBinSize) + 1)
        nexttile;
        hold on
        plotResults(all_trials_phase, trial_outcome, "r", .9, j, frequency)
        plotResults(all_trials_phase, ~trial_outcome, "b", .8, j, frequency)
    end
    
end



function times = convertTimes(z, y)
    times = min(find(abs(z - y) < .1));
    if isempty(times)
        times = NaN;
        "couldn't find time"
    end 
end

function trialed = trialData(start, data, recordingFrequency, fftLength, post)
    if nargin < 5
        post = true;
    end
    if post
        if  start + str2num(recordingFrequency) * (fftLength - 1) > length(data)
            trialed = data(:,start- str2num(recordingFrequency):length(data));% - str2num(recordingFrequency));
            return
        end
        trialed = data(:,start- str2num(recordingFrequency):start + str2num(recordingFrequency) * (fftLength - 1));% - str2num(recordingFrequency));
    else
        if  start - str2num(recordingFrequency) * (fftLength + 1) < 0
            trialed = data(:,0:start - str2num(recordingFrequency));% - str2num(recordingFrequency));
            return
        end
        trialed = data(:,start - str2num(recordingFrequency) * (fftLength + 1):start - str2num(recordingFrequency));% - str2num(recordingFrequency));
    end
    
end
function [] = plotResults(all_trials_phase, trial_outcome, color, radius, sub, frequency)
    %trial_outcome = trial_outcome(1:end - 1);  %for stim onset
    % figure(1)
    ang = angle(all_trials_phase(find(trial_outcome),:,frequency:frequency));
    
    %histogram(ang, 'BinEdges',[-pi, -pi * 3/4, -pi * 1/2, -pi * 1/4, 0, pi * 1/4, pi * 1/2 , pi * 3/4, pi])
    

    mag = abs(all_trials_phase(:,:,frequency:frequency));
    average_mag = mean(mag, [1,2]);
    [~, maxInd] = max(average_mag);
    if color == "r"
        maxInd
    end
    


    unit_vec = cos(ang) + 1i * sin(ang);
    unit_vec = squeeze(mean(unit_vec,2));
    
    average_phase = angle(unit_vec);
    writematrix(average_phase, "StimOffset/" +int2str(sub) + '-promptPhase.csv');
    unit_vec = cos(average_phase) + 1i * sin(average_phase);
    
    
    plot(unit_vec(:,maxInd), ".",'MarkerSize',20, "MarkerEdgeColor", color)

    average_vector = squeeze(mean(unit_vec, 1));
    average_phase = angle(average_vector);
    average_unit = cos(average_phase) + 1i * sin(average_phase);
    plot(average_unit(maxInd), ".",'MarkerSize',50, "MarkerEdgeColor", color)

    std_vector = sqrt(-2*log(abs(average_vector)));
    %errorbar(average_phase(2), std_vector(2))


    circr = @(radius,rad_ang)  [radius*cos(rad_ang);  radius*sin(rad_ang)];         % Circle Function For Angles In Radians
    N = 25;                                                         % Number Of Points In Complete Circle
    r_angl = linspace(average_phase(maxInd) - std_vector(maxInd), average_phase(maxInd) + std_vector(maxInd), N);                             % Angle Defining Arc Segment (radians)
    xy_r = circr(radius,r_angl);             
    % Matrix (2xN) Of (x,y) Coordinates
    % figure
    plot(xy_r(1,:), xy_r(2,:), 'bp', "MarkerEdgeColor", color)                                % Draw An Arc 
    axis equal
    xlim([-1 1])
    ylim([-1 1])
    % savefig(int2str(sub))
end