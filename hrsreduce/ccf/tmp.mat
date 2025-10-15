function [radial_velocity,rv_error,systemic_velocity_direct,systemic_velocity_gauss,vsini,vsini_error,vsini_fit]=radial_velocity_calculation(coarse_ccf)

%determine extent of profile
summed_line=sum(coarse_ccf.intensity,1);
[pks, locs, w]=findpeaks(max(summed_line)-summed_line,coarse_ccf.velocity,'Annotate','extents','MinPeakProminence',max(max(summed_line)-summed_line)/2,'WidthReference','halfheight');
[negpks, neglocs]= findpeaks(summed_line,coarse_ccf.velocity);
try
minidx = find(neglocs < locs-w/1.2, 1, 'last');
catch
    radial_velocity=NaN;
    rv_error=NaN;
    systemic_velocity_direct=NaN;
    systemic_velocity_gauss=NaN;
    vsini=NaN;
    vsini_error=NaN;
    vsini_fit=NaN
    return
end

left_sym_flag=0;
right_sym_flag=0;

%left shoulder
if isempty(minidx)
    try
        minloc=neglocs(1);
        if minloc>locs(1)
            minloc=coarse_ccf.velocity(1);
            left_sym_flag=1;
        end
    catch
        minloc=coarse_ccf.velocity(1);
        left_sym_flag=1;
    end
else
    minloc = neglocs(minidx);
end
%right shoulder
maxidx = find(neglocs > locs+w/1.2, 1, 'first');

if isempty(maxidx)
    try
    maxloc=neglocs(end);
    if maxloc<locs(1)
            maxloc=coarse_ccf.velocity(end);
            right_sym_flag=1;
        end
    catch
        maxloc=coarse_ccf.velocity(end);
        right_sym_flag=1;
    end
else 
    maxloc = neglocs(maxidx);
end

%make profile symmetric if using edges
if left_sym_flag==1 && right_sym_flag==0
    sym=abs(maxloc-locs(1));
    minloc=locs(1)-sym;
end

if left_sym_flag==0 && right_sym_flag==1
    sym=abs(locs(1)-minloc);
    maxloc=locs(1)+sym;
end    
    

%dealing with a profile on the edge
if minloc==maxloc
    if abs(coarse_ccf.velocity(1)-minloc)<abs(coarse_ccf.velocity(end)-minloc)
        minloc=coarse_ccf.velocity(1);
    else
        maxloc=coarse_ccf.velocity(end);
    end
end

cut_line_velocity=coarse_ccf.velocity(find_nearest(coarse_ccf.velocity,minloc):find_nearest(coarse_ccf.velocity,maxloc));
delta_velocity=abs(cut_line_velocity(2)-cut_line_velocity(1));

%small moments
for n=1:length(coarse_ccf.jd)
    intensity=coarse_ccf.intensity(n,find_nearest(coarse_ccf.velocity,minloc):find_nearest(coarse_ccf.velocity,maxloc));
    moment_1=0;
    moment_0=0;
    for m=1:numel(cut_line_velocity)
        moment_1=moment_1+((1-intensity(m))*(cut_line_velocity(m))*delta_velocity);
        moment_0=moment_0+((1-intensity(m))*delta_velocity);
    end
    normalising_factor_values(n)=moment_1/moment_0;
    unnormalised_moment_1(n)=moment_1;
    unnormalised_moment_0(n)=moment_0;
    %vsini
    %least-squares fit needs the wings to be at 1
    botpix=find_nearest(coarse_ccf.velocity,minloc);
    toppix=find_nearest(coarse_ccf.velocity,maxloc);
    bkgrd=mean([coarse_ccf.intensity(n,1:(botpix-10)) coarse_ccf.intensity(n,(toppix+10):end)]);
    try
    [vsini(n),fittedint]=vsini_directfit(coarse_ccf.velocity,coarse_ccf.intensity(n,:)./bkgrd,locs);
    [v_gpks(n),v_glocs(n),v_gwidths(n),v_gproms(n),bPk,iLB,iRB]=findpeaks_vsini(1-fittedint,coarse_ccf.velocity,'Annotate','extents','MinPeakProminence',max(1-fittedint)/10,'WidthReference','halfprom');
    catch
        vsini(n)=NaN;
        vsini_error(n)=NaN;
        v_glocs(n)=NaN;
        continue
    end
    %vsini_errors
    vsini_error(n)=(delta_velocity);
end

normalising_factor=mean(normalising_factor_values);
mom_0_error=delta_velocity;
mom_1_error=sqrt((delta_velocity/2)^2+delta_velocity^2);
single_norm_factor_error=sqrt(mom_0_error^2+mom_1_error^2);
norm_factor_error=(single_norm_factor_error)/sqrt(numel(cut_line_velocity));

%large moments
for p=1:length(coarse_ccf.jd)
    intensity=coarse_ccf.intensity(p,find_nearest(coarse_ccf.velocity,minloc):find_nearest(coarse_ccf.velocity,maxloc));
    Moment_1=0;
    Moment_0=0;
    for q=1:numel(cut_line_velocity)
        Moment_1=Moment_1+((1-intensity(q))*(cut_line_velocity(q)-normalising_factor)*delta_velocity);
        Moment_0=Moment_0+((1-intensity(q))*delta_velocity);
    end
    radial_velocity(p)=Moment_1/Moment_0;
    unnormalised_Moment_1(p)=Moment_1;
    unnormalised_Moment_0(p)=Moment_0;
end

Mom_0_error=delta_velocity;
Mom_1_error=sqrt((delta_velocity/2+norm_factor_error)^2+delta_velocity^2);
rv_error=sqrt(Mom_0_error^2+Mom_1_error^2)/sqrt(numel(cut_line_velocity));
systemic_velocity_direct=locs;
systemic_velocity_gauss=nanmedian(v_glocs);
try
    vsini_fit=w/2;
catch
    vsini_fit=NaN;
end