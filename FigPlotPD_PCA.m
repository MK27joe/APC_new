function FigPlotPD_PCA(T1,TankNum,IndX,FaultID,lim,Dslope,Mag_PD,ts1,TS,SPE,S1,Dtest,e)

    figure(1)

    subplot(2,1,1)
    % figure(1)
    set(gcf, 'WindowState', 'maximized');
    plot(ts1, 'b','LineWidth',2); 
    ylabel('PCA-T^2','fontsize',14,'FontName', 'Arial');
    hold on;
    plot(TS,'r--', 'LineWidth',3);  
    xlabel('Observation Number','fontsize',12,'FontName', 'Arial');
    xlim([0 size(Dtest,1)]);
    title(['FD for ', num2str(T1),'-Tank system: T^2 limits with SNR = NA and Fault = ', FaultID,'; of Age-Slope = ', num2str(Dslope), '; Degradation = ', num2str(Mag_PD)]);
    subtitle(['Fault at n = ', num2str(lim),' for Tank # ', num2str(TankNum)]);
    legend({'TS','TS-Thres.'},'Location', 'best'),
    grid


    subplot(2,1,2)

    set(gcf, 'WindowState', 'maximized');
    plot(SPE,'k','LineWidth',2); 
    xlim([0 size(Dtest,1)]);
    %ylim([-50000 290000])
    ylabel('PCA-Q','fontsize',14,'FontName', 'Arial');
    hold on;
    plot(S1,'m--', 'LineWidth',3);  
    xlabel('Observation Number','fontsize',12,'FontName', 'Arial');
    title(['FD for ', num2str(T1),'-Tank system: SPE/Q limits with SNR = NA and Fault = ', FaultID,'; of Age-Slope = ', num2str(Dslope), '; Degradation = ', num2str(Mag_PD)]);
    subtitle(['Fault at n = ', num2str(lim),' for Tank # ', num2str(TankNum)]);
    legend({'SPE','Q-Thres.'},'Location', 'best')
    grid;

    filename1 = "FD_" + num2str(T1) + "T_t" + num2str(TankNum) + "_" + FaultID + "-AS" + num2str(Dslope) + "-D" + num2str(Mag_PD) + "_" + string(datetime('now'),'yy-MM-dd_HH_mm_ss')+".png";
    saveas(gcf,filename1); 

    %%%------%%


    f2 = figure(2); % Plot Test Dataset 
    set(f2,'Position',get(0,'screensize'));
    plot(Dtest,'LineWidth',2)
    xlim([0 size(Dtest,1)])
    xlabel('Observation Number','fontsize',12,'FontName', 'Arial');
    ylabel('Dataset Vectors','fontsize',14,'FontName', 'Arial');
    title(['Test dataset vectors for ', num2str(T1),'-Tank system [SNR = NA and Fault = ', FaultID,'; of Age-Slope = ', num2str(Dslope), '; Degradation = ', num2str(Mag_PD),']']);
    subtitle(['Fault at n = ', num2str(lim),' for Tank # ', num2str(TankNum)]);
    legend({'V1','V2','h1','h2','h3','h4'},'location','best')
    grid;

    filename2 = "Dtest_" + num2str(T1) + "T_t" + num2str(TankNum) + "_" + FaultID + "-AS" + num2str(Dslope) + "-D" + num2str(Mag_PD) + "_" + string(datetime('now'),'yy-MM-dd_HH_mm_ss')+".png";
    saveas(gcf,filename2); 

    %%%------%%%


    f3 = figure(3); % Plot the complete residue space
    set(f3,'Position',get(0,'screensize'));
    plot(e,'LineWidth',2)
    xlim([0 size(Dtest,1)])
    xlabel('Observation Number','fontsize',12,'FontName', 'Arial');
    ylabel('Error Values','fontsize',14,'FontName', 'Arial');
    title(['Model Errors for ', num2str(T1),'-Tank system [SNR = NA and Fault = ', FaultID,'; of Age-Slope = ', num2str(Dslope), '; Degradation = ', num2str(Mag_PD),']']);
    subtitle(['Fault at n = ', num2str(lim),' for Tank # ', num2str(TankNum)]);
    legend({'V1','V2','h1','h2','h3','h4'},'location','best')
    grid;

    filename3 = "Residues_" + num2str(T1) + "T_t" + num2str(TankNum) + "_" + FaultID + "-AS" + num2str(Dslope) + "-D" + num2str(Mag_PD) + "_" + string(datetime('now'),'yy-MM-dd_HH_mm_ss')+".png";
    saveas(gcf,filename3); 

    %%%-----%%%

    f4 = figure(4); % Plot the faulty test-data vector
    set(f4,'Position',get(0,'screensize'));
    plot(e(:,IndX),'k','LineWidth',2)
    xlim([0 size(Dtest,1)])
    xlabel('Observation Number','fontsize',12,'FontName', 'Arial');
    ylabel('Error Values','fontsize',14,'FontName', 'Arial');
    title(['Error Plot (Faulty vector) for ', num2str(T1),'-Tank system [SNR = NA and Fault = ', FaultID,'; of Age-Slope = ', num2str(Dslope), '; Degradation = ', num2str(Mag_PD),']']);
    subtitle(['Fault at n = ', num2str(lim),' for Tank # ', num2str(TankNum)]);
    grid;

    filename4 = "FaultyVect_" + num2str(T1) + "T_t" + num2str(TankNum) + "_" + FaultID + "-AS" + num2str(Dslope) + "-D" + num2str(Mag_PD) + "_" + string(datetime('now'),'yy-MM-dd_HH_mm_ss')+".png";
    saveas(gcf,filename4);


end