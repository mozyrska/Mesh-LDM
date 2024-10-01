function x = comp_percentile(datas,value)
    perc = [0:0.1:100];
    perc_vals = prctile(datas, perc);
    [c index] = min(abs(perc_vals'-value));
    x = perc(index);
end
