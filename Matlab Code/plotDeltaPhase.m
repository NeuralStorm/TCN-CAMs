%% Delta phase across trials
% represented as a heatmap, as well as a trial-avg time series for each
% subject
%% Folder and file organization
% Set up parameters, scripts, FieldTrip defaults
subject_inc = [2,3,4,5,6,7,8,9,10,13,14,15,16,18,19];
peak_deltas = [3 2 3 3 4 4 4 2 2 3 4 2 3 3 3 4];
load("subj_trial_meta.mat")
fields = fieldnames(TrialData);
fs = 1000;
t_idx = 4000;
plvs = [];
cplvs = [];

for j = 1:15
    if subject_inc(j) < 10
        SUBID = ['s0'  int2str(subject_inc(j))];
        load(['data/s0', int2str(subject_inc(j)), '_GTH_decision_data_preproc14.mat'])
        load(['data/GTH_s0', int2str(subject_inc(j)), '_decision_power_struct_nobs.mat'])
    else
        SUBID = ['s'  int2str(subject_inc(j))];
        load(['data/s', int2str(subject_inc(j)), '_GTH_decision_data_preproc14.mat'])
        load(['data/GTH_s', int2str(subject_inc(j)), '_decision_power_struct_nobs.mat'])
    end
    tic
   
    % behavioral data trialing
    gamble_trials = power_struct.beh.gambles;
    safe_trials = power_struct.beh.safebet;

    %% estimate delta phase over time
    % filter-hilbert method

    % delta peak
    cfg = [];
    cfg.bpfilter = 'yes';
    cfg.bpfreq = [peak_deltas(j) - .5, peak_deltas(j) + .5];
    cfg.bpfilttype = 'fir';
    %cfg.hilbert = 'angle';
    %cfg.trials = gamble_trials;
    data_delta = ft_preprocessing(cfg, data_tr_pp_ds);

    % averaged delta
    delta_average = data_delta;
    delta_average.trial = {};
    delta_average.trial = cellfun(@(x) mean(x, 'omitmissing'), data_delta.trial, 'UniformOutput', false);
    delta_average.label = {'average'};
    
    % now calculate phase
    cfg = [];
    cfg.bpfilter = 'no';
    cfg.hilbert = 'angle';
    %cfg.trials = TrialData.(SUBID).correct_gamble;
    data_phase = ft_preprocessing(cfg, data_delta);
    % change angle range from 0 to 2pi
    
    % select for only the second before choice
    cfg = [];
    cfg.toilim = [-1, 0];
    data_phase_trimmed = ft_redefinetrial(cfg,data_phase);

    all_data = cat(3, data_phase_trimmed.trial{:});
    all_data_mod = mod(all_data + 2 * pi, 2 * pi);

    % reorder based on trial length
    
    [B, sort_trials] = sort(TrialData.(SUBID).response_times,'ascend');
    all_data_mod_sorted = all_data_mod(:,:,sort_trials);

    
    %% create a scrambled control with randomized choice time
    % choice is randomly assigned at somewhere within the second prior to
    % choice
    random_shift = (rand(size(all_data_mod_sorted,3),1))*data_phase.fsample;
    cfg = [];
    cfg.offset = random_shift;
    data_scrambled = ft_redefinetrial(cfg,data_phase);

    cfg = [];
    cfg.toilim = [-1 0];
    data_scrambled_trimmed = ft_redefinetrial(cfg,data_scrambled);

    all_data_s = cat(3, data_scrambled_trimmed.trial{:});
    all_data_s_mod = mod(all_data_s + 2 * pi, 2 * pi);

    % reorder based on trial length
    [B, sort_trials] = sort(TrialData.(SUBID).response_times,'ascend');
    all_data_s_mod_sorted = all_data_s_mod(:,:,sort_trials);


    %% for each subject, create a histogram of the delta phase at choice
    % all_correct = [find(TrialData.(SUBID).correct_safe);find(TrialData.(SUBID).correct_gamble)];
    % all_mistake = [find(TrialData.(SUBID).mistake_safe);find(TrialData.(SUBID).mistake_gamble)];
    % hist_values = squeeze(mean(all_data_mod(:,end,all_correct),1));
    % mistake_values = squeeze(mean(all_data_mod(:,end,all_mistake),1));
    % f = figure;
    % histogram(hist_values);
    % hold on
    % histogram(mistake_values);



    %% create figures
    mkdir([SUBID,'\all_trials'])
    for chan_idx = 1:numel(data_phase.label)
        plot_data = squeeze(all_data_mod_sorted(chan_idx,:,:));
        cf = figure;
        imagesc(plot_data');
        ylabel('Trials','FontSize',7,FontName='Arial');
        xlabel('Time to choice (ms)','FontSize',7,FontName='Arial');

        colormap(crameri('romaO'));
        xline(1000, 'r--','DisplayName','button press','LineWidth',3,'LabelHorizontalAlignment','left');
        set(gca, 'YDir','normal');
        % Create textbox
        % title(sprintf('%s channel %s delta phase across trials',SUBID,data_phase.label{chan_idx}),'FontSize',7,FontName='Arial');
        % add button press label
        % Create legend
        %legend1 = legend(gca,'show');
        %set(legend1,...
        %'Position',[0.566705060877335 0.862713647894117 0.210714282148651 0.0476190464837211],...
        %'Orientation','horizontal');
         width_in = 3.7139;
    height_in = 2.8889;
    set(cf, 'Units', 'inches');               % Set units to centimeters
    set(cf, 'Position', [5, 5, width_in, height_in]); % Set [left, bottom, width, height]
        set(gcf,'Color','w');

        ax = gca;
        ax.YAxis.FontSize = 7; %for y-axis 
        ax.XAxis.FontSize = 7; %for y-axis 
        ax.XColor = 'k';
        ax.YColor = 'k';
        fontname('Arial')

        cb = colorbar;
        ylabel(cb,'Delta phase','FontSize',14)
        cb.Ticks = [0, pi/2, pi 3*pi/2, 2*pi]; 
        cb.TickLabels = {'0','π/2','π','3π/2','2π'};
        cb.FontSize = 7;
        cb.FontName = 'Arial';
        clim([-0.0001, 6.3]);

        xt = [0 get(gca, 'XTick')];
        set(gca, 'XTickLabel', fliplr(xt(1:end-1)));
        % save each channel
        set(gcf,'Renderer','painters');
        saveas(gcf,[SUBID,'\all_trials\',data_phase.label{chan_idx},'.png']);
        exportgraphics(gcf,[SUBID,'\all_trials\',data_phase.label{chan_idx},'.pdf'],'ContentType','vector');
        close;


        %% plot scrambled_data
        plot_data_s = squeeze(all_data_s_mod_sorted(chan_idx,:,:));
        cf_s = figure;
        imagesc(plot_data_s');
        ylabel('Trials','FontSize',7,FontName='Arial');
        xlabel('Time to choice (ms)','FontSize',7,FontName='Arial');

        colormap(crameri('romaO'));
        xline(1000, 'r--','DisplayName','button press','LineWidth',3,'LabelHorizontalAlignment','left');
        set(gca, 'YDir','normal');
        % Create textbox
        %title(sprintf('%s channel %s delta phase across trials CONTROL',SUBID,data_phase.label{chan_idx}),'FontSize',14,'FontWeight','bold',FontName='Arial');
        % add button press label
        % Create legend
        %legend1 = legend(gca,'show');
        %set(legend1,...
        %'Position',[0.566705060877335 0.862713647894117 0.210714282148651 0.0476190464837211],...
        %'Orientation','horizontal');
                 width_in = 3.7139;
    height_in = 2.8889;
    set(cf_s, 'Units', 'inches');               % Set units to centimeters
    set(cf_s, 'Position', [5, 5, width_in, height_in]); % Set [left, bottom, width, height]
        set(gcf,'Color','w');

        ax = gca;
        ax.YAxis.FontSize = 7; %for y-axis 
        ax.XAxis.FontSize = 7; %for y-axis 
        ax.XColor = 'k';
        ax.YColor = 'k';
        fontname('Arial')

        cb = colorbar;
        ylabel(cb,'Delta phase','FontSize',14)
        cb.Ticks = [0, pi/2, pi 3*pi/2, 2*pi]; 
        cb.TickLabels = {'0','π/2','π','3π/2','2π'};
        clim([-0.0001, 6.3]);
        cb.FontSize = 7;
        cb.FontName = 'Arial';

        xt = [0 get(gca, 'XTick')];
        set(gca, 'XTickLabel', fliplr(xt(1:end-1)));
     
        % save each channel
        %set(gcf,'Renderer','painters');
        saveas(gcf,[SUBID,'\all_trials\',data_phase.label{chan_idx},'_scramble','.png']);
        exportgraphics(gcf,[SUBID,'\all_trials\',data_phase.label{chan_idx},'_scramble','.pdf'],'ContentType','vector');
        close
    end
    close all
end

xl = xlim; yl = ylim; zl = zlim;
set(gca, 'XTick', linspace(xl(1), xl(2), 3), ...
         'YTick', linspace(yl(1), yl(2), 3), ...
         'ZTick', linspace(zl(1), zl(2), 3));
title(get(gca,'Title').String, 'FontSize', 13, 'FontName', 'Arial');


