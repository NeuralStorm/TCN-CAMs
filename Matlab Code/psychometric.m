f = figure;
T = tiledlayout(3, 5,"TileSpacing","loose", "Padding", "loose");
subjs = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16, 18, 19];
load("subj_trial_meta.mat")

risk = zeros(15,1);

for j = 1:15
    sub = subjs(j);
    if sub < 10
        load("data/GTH_s0" + sub + "_decision_power_struct_nobs.mat");
        field = "s0" + string(sub)
    else
        load("data/GTH_s" + sub + "_decision_power_struct_nobs.mat");
        field = "s" + string(sub)
    end
    
    probs = TrialData.(field).win_probs;
    data = zeros(10,1);
    for i = 0:10
        data(i + 1) = mean(power_struct.beh.gambles(abs(probs - (i * .1)) < .0001));
    end
    nexttile

    xdata = 0:.1:1;
    ydata = data;

    ft = fittype('1/(1+exp(-k*(x-x0)))','independent','x','coefficients',{'k','x0'});
    f = fit(xdata', ydata, ft, 'StartPoint', [10 0.5]);
    risk(j) = f(.5);
    plot(f, xdata, ydata)
end