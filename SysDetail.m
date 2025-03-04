
function SysDetail()


    disp('Input Vectors:'); disp('``````````````````')
    fprintf('Motor input voltages:\nV1, V2, V3 and V4\n\n')
    disp('Output Vectors:'); disp('``````````````````')
    fprintf('Fluid levels in:\n')
    disp('Tank # 1 --> H1 --> ID: t8h1');
    disp('Tank # 2 --> H2 --> ID: t8h2');
    disp('Tank # 3 --> H3 --> ID: t8h3');
    disp('Tank # 4 --> H4 --> ID: t8h4');
    disp('Tank # 5 --> H5 --> ID: t8h5');
    disp('Tank # 6 --> H6 --> ID: t8h6');
    disp('Tank # 7 --> H7 --> ID: t8h7');
    disp('Tank # 8 --> H8 --> ID: t8h8');
    %
    fprintf('\n\nInitial Condition:\n'), disp('^^^^^^^^^^^^^^^^^^^^^^'),fprintf('\n')
    disp('@INPUT:')
    fprintf('\nV10 = %0.4f', T_initial(1,1))
    fprintf('\nV20 = %0.4f', T_initial(1,2))
    fprintf('\nV30 = %0.4f', T_initial(1,3))
    fprintf('\nV40 = %0.4f\n\n', T_initial(1,4))

    disp('@OUTPUT:')
    fprintf('\nH10 = %0.4f', T_initial(1,5))
    fprintf('\nH20 = %0.4f', T_initial(1,6))
    fprintf('\nH30 = %0.4f', T_initial(1,7))
    fprintf('\nH40 = %0.4f', T_initial(1,8))
    fprintf('\nH50 = %0.4f', T_initial(1,9))
    fprintf('\nH60 = %0.4f', T_initial(1,10))
    fprintf('\nH70 = %0.4f', T_initial(1,11))
    fprintf('\nH80 = %0.4f\n\n', T_initial(1,12))

end