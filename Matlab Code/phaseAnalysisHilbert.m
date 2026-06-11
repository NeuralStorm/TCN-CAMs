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

TPlot = tiledlayout(4, 4,"TileSpacing","loose", "Padding", "loose");
rs = [];
ks = [];
for j = 6:21
    control = false;
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
    eeg_control_indices = round(min(eeg_start_indices) + rand(200,1) * (max(eeg_start_indices) - min(eeg_start_indices)));
    
    size(eeg_start_indices)
    fs = str2num(loadedData.eeg.lslInfo.nominal_srate);
    bpFilt = designfilt('bandpassiir', ...
        'FilterOrder', 10, ...
        'HalfPowerFrequency1', .5, ...
        'HalfPowerFrequency2', 1.5, ...
        'SampleRate', fs);
    
    
    % filter along time (dimension 2)
    filtered = filtfilt(bpFilt, loadedData.eeg.data')';  % keep channels x time
    
    % Hilbert transform → analytic signal
    analytic = hilbert(filtered')';  
    
    % instantaneous phase at decision
    phi_t = angle(analytic(:, eeg_start_indices));

    control_phi = angle(analytic(:, eeg_control_indices));
    
    C = mean(cos(phi_t));
    S = mean(sin(phi_t));
    circ_mean = atan2(S, C);

    C_control = mean(cos(control_phi));
    S_control = mean(sin(control_phi));
    control_circ_mean = atan2(S_control, C_control);
    
    C = mean(cos(phi_t(1,:)));
    S = mean(sin(phi_t(1,:)));
    R = sqrt(C^2 + S^2);
    circ_std = sqrt(-2 * log(R));
    
    rs = [rs circ_rtest(circ_mean)];
    ks = [ks circ_kuipertest(circ_mean, control_circ_mean)];
    nexttile
    p = polarscatter(circ_mean,ones(1,200), 50, 'filled', 'MarkerFaceColor', 'b');
    p.MarkerFaceAlpha = 0.3;
end


function times = convertTimes(z, y)
    times = min(find(abs(z - y) < .1));
    if isempty(times)
        times = NaN;
        "couldn't find time"
    end 
end
