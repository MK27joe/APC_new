function FigPlotBias_PCA(T1,TankNum,IndX,FaultID,Bias,lim,ts1,TS,SPE,S1,Dtest,e)
    
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
    title(['FD for ', num2str(T1),'-Tank system: T^2 limits for SNR = [] and Fault = ', FaultID,' of value = ', num2str(Bias)]);
    subtitle(['Fault at n = ', num2str(lim),' for Tank # ', num2str(TankNum)]);
    legend({'TS','TS-Thres.'},'Location', 'best'),
    grid

    
    subplot(2,1,2)
    
    set(gcf, 'WindowState', 'maximized');

    plot(SPE,'k','LineWidth',2); 
    
    xlim([0 size(Dtest,1)]);
    %ylim([-50000 290000])
    ylabel('PCA-SPE-Q','fontsize',14,'FontName', 'Arial');
    hold on;
    plot(S1,'m--', 'LineWidth',3);  
    xlabel('Observation Number','fontsize',12,'FontName', 'Arial');
    title(['FD for ', num2str(T1),'-Tank system: SPE/Q limits for SNR = [] and Fault = ', FaultID,' of value = ', num2str(Bias)]);
    subtitle(['Fault at n = ', num2str(lim),' for Tank # ', num2str(TankNum)]);
    legend({'SPE','Q-Thres.'},'Location', 'best')
    grid;

    % filename1 = "FD_" + num2str(T1) + "T_t" + num2str(TankNum) + "_" + FaultID + "-" + num2str(Bias) + "_" + string(datetime('now'),'yy-MM-dd_HH_mm_ss')+".png";
    % saveas(gcf,filename1); 

    %%%------%%


    f2 = figure(2); % Plot Test Dataset 
    set(f2,'Position',get(0,'screensize'));

    plot(Dtest(:,1),'Color',[1 0 0],'Marker','.','LineStyle', '-','LineWidth',1); grid, hold on;
    plot(Dtest(:,2),'Color',[0 0 1],'Marker','.','LineStyle', '-','LineWidth',1); grid, hold on;
    plot(Dtest(:,3),'Color',[0 0 0],'Marker','.','LineStyle', '-','LineWidth',1); grid, hold on;
    plot(Dtest(:,4),'Color',[1 0 1],'Marker','.','LineStyle', '-','LineWidth',1); grid; hold on;

    plot(Dtest(:,5),'Color',[0 0.4470 0.7410],'Marker','.','LineStyle', '-','LineWidth',2); grid; hold on;
    plot(Dtest(:,6),'Color',[0.8500 0.3250 0.0980],'Marker','.','LineStyle', '-','LineWidth',2); grid; hold on;
    plot(Dtest(:,7),'Color',[0 1 0],'Marker','.','LineStyle', '-','LineWidth',2); grid; hold on;
    plot(Dtest(:,8),'Color',[0.9290 0.6940 0.1250],'Marker','.','LineStyle', '-','LineWidth',2); grid; hold on;
    plot(Dtest(:,9),'Color',[0.4940 0.1840 0.5560],'Marker','.','LineStyle', '-','LineWidth',2); grid; hold on;
    plot(Dtest(:,10),'Color',[0.4660 0.6740 0.1880],'Marker','.','LineStyle', '-','LineWidth',2); grid; hold on;
    plot(Dtest(:,11),'Color',[0.3010 0.7450 0.9330],'Marker','.','LineStyle', '-','LineWidth',2); grid; hold on;
    plot(Dtest(:,12),'Color',[0.6350 0.0780 0.1840],'Marker','.','LineStyle', '-','LineWidth',2); grid; 


    xlim([0 size(Dtest,1)])
    xlabel('Observation Number','fontsize',12,'FontName', 'Arial');
    ylabel('Dataset Vectors','fontsize',14,'FontName', 'Arial');
    title(['Test dataset vectors for ', num2str(T1),'-Tank system [SNR = [] and Fault = ', FaultID,' and of value = ', num2str(Bias),']']);
    subtitle(['Fault at n = ', num2str(lim),' for Tank # ', num2str(TankNum)]);
    legend({'h1','h2','h3','h4','h5','h6','h7','h8','V1','V2','V3','V4'},'location','bestoutside')
    grid;

    % filename2 = "Dtest_" + num2str(T1) + "T_t" + num2str(TankNum) + "_" + FaultID + "-" + num2str(Bias) + "_" + string(datetime('now'),'yy-MM-dd_HH_mm_ss')+".png";
    % saveas(gcf,filename2); 




    %%%------%%%


    f3 = figure(3); % Plot the complete residue space
    set(f3,'Position',get(0,'screensize'));

    plot(e(:,1),'Color',[1 0 0],'Marker','.','LineStyle', '-','LineWidth',1); grid, hold on;
    plot(e(:,2),'Color',[0 0 1],'Marker','.','LineStyle', '-','LineWidth',1); grid, hold on;
    plot(e(:,3),'Color',[0 0 0],'Marker','.','LineStyle', '-','LineWidth',1); grid, hold on;
    plot(e(:,4),'Color',[1 0 1],'Marker','.','LineStyle', '-','LineWidth',1); grid; hold on;

    plot(e(:,5),'Color',[0 0.4470 0.7410],'Marker','.','LineStyle', '-','LineWidth',2); grid; hold on;
    plot(e(:,6),'Color',[0.8500 0.3250 0.0980],'Marker','.','LineStyle', '-','LineWidth',2); grid; hold on;
    plot(e(:,7),'Color',[0 1 0],'Marker','.','LineStyle', '-','LineWidth',2); grid; hold on;
    plot(e(:,8),'Color',[0.9290 0.6940 0.1250],'Marker','.','LineStyle', '-','LineWidth',2); grid; hold on;
    plot(e(:,9),'Color',[0.4940 0.1840 0.5560],'Marker','.','LineStyle', '-','LineWidth',2); grid; hold on;
    plot(e(:,10),'Color',[0.4660 0.6740 0.1880],'Marker','.','LineStyle', '-','LineWidth',2); grid; hold on;
    plot(e(:,11),'Color',[0.3010 0.7450 0.9330],'Marker','.','LineStyle', '-','LineWidth',2); grid; hold on;
    plot(e(:,12),'Color',[0.6350 0.0780 0.1840],'Marker','.','LineStyle', '-','LineWidth',2); grid; 
    
    xlim([0 size(Dtest,1)])
    xlabel('Observation Number','fontsize',12,'FontName', 'Arial');
    ylabel('Error Values','fontsize',14,'FontName', 'Arial');
    title(['Model Errors for ', num2str(T1),'-Tank system [SNR = [] and Fault = ', FaultID,' and of value = ', num2str(Bias),']']);
    subtitle(['Fault at n = ', num2str(lim),' for Tank # ', num2str(TankNum)]);
    legend({'h1','h2','h3','h4','h5','h6','h7','h8','V1','V2','V3','V4'},'location','bestoutside')
    grid;

    % filename3 = "Residues_" + num2str(T1) + "T_t" + num2str(TankNum) + "_" + FaultID + "-" + num2str(Bias) + "_" + string(datetime('now'),'yy-MM-dd_HH_mm_ss')+".png";
    % saveas(gcf,filename3); 



    %%%-----%%%

    f4 = figure(4); % Plot the faulty test-data vector
    set(f4,'Position',get(0,'screensize'));

    plot(e(:,IndX),'k','LineWidth',2)

    xlim([0 size(Dtest,1)])
    xlabel('Observation Number','fontsize',12,'FontName', 'Arial');
    ylabel('Error Values','fontsize',14,'FontName', 'Arial');
    title(['Error Plot (Faulty vector) for ', num2str(T1),'-Tank system [SNR = [] and Fault = ', FaultID,' and of value = ', num2str(Bias),']']);
    subtitle(['Fault at n = ', num2str(lim),' for Tank # ', num2str(TankNum)]);
    grid;

    % filename4 = "FaultyVect_" + num2str(T1) + "T_t" + num2str(TankNum) + "_" + FaultID + "-" + num2str(Bias) + "_" + string(datetime('now'),'yy-MM-dd_HH_mm_ss')+".png";
    % saveas(gcf,filename4);


end

 
