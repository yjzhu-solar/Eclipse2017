pro cal_pop_proc

    streamer_file = "../sav/AWSoM/streamer_example.csv"
    streamer_data = read_csv(streamer_file,header=steamer_header)
    help,streamer_data

    r_distance = streamer_data.field1 - 0.05
    streamer_dens = streamer_data.field2
    streamer_temp = streamer_data.field3

    n_points = n_elements(r_distance)

    FeXIV_out = dblarr(n_points,8)
    FeXIV_in = dblarr(n_points,9)
    FeX_out = dblarr(n_points,8)
    FeX_in = dblarr(n_points,9)

    for ii = 0,n_points - 1 do begin
        print,r_distance[ii],alog10(streamer_dens[ii]),alog10(streamer_temp[ii])
        pop_processes, "fe_10",dens=streamer_dens[ii],temp=streamer_temp[ii], $
                    level=2,rphot=r_distance[ii],radtemp=5770d,output=output_FeX
        pop_processes, "fe_14",dens=streamer_dens[ii],temp=streamer_temp[ii], $
                    level=2,rphot=r_distance[ii],radtemp=5770d,output=output_FeXIV

        FeX_out[ii,0] = output_FeX.out.rad_decay
        FeX_out[ii,1] = output_FeX.out.e_exc
        FeX_out[ii,2] = output_FeX.out.e_deexc
        FeX_out[ii,3] = output_FeX.out.p_exc
        FeX_out[ii,4] = output_FeX.out.p_deexc
        FeX_out[ii,5] = output_FeX.out.ph_exc
        FeX_out[ii,6] = output_FeX.out.ph_deexc
        FeX_out[ii,7] = output_FeX.out.ai

        FeX_in[ii,0] = output_FeX.in.rad_decay
        FeX_in[ii,1] = output_FeX.in.e_exc
        FeX_in[ii,2] = output_FeX.in.e_deexc
        FeX_in[ii,3] = output_FeX.in.p_exc
        FeX_in[ii,4] = output_FeX.in.p_deexc
        FeX_in[ii,5] = output_FeX.in.ph_exc
        FeX_in[ii,6] = output_FeX.in.ph_deexc
        FeX_in[ii,7] = output_FeX.in.rr
        FeX_in[ii,8] = output_FeX.in.dc

        FeXIV_out[ii,0] = output_FeXIV.out.rad_decay
        FeXIV_out[ii,1] = output_FeXIV.out.e_exc
        FeXIV_out[ii,2] = output_FeXIV.out.e_deexc
        FeXIV_out[ii,3] = output_FeXIV.out.p_exc
        FeXIV_out[ii,4] = output_FeXIV.out.p_deexc
        FeXIV_out[ii,5] = output_FeXIV.out.ph_exc
        FeXIV_out[ii,6] = output_FeXIV.out.ph_deexc
        FeXIV_out[ii,7] = output_FeXIV.out.ai

        FeXIV_in[ii,0] = output_FeXIV.in.rad_decay
        FeXIV_in[ii,1] = output_FeXIV.in.e_exc
        FeXIV_in[ii,2] = output_FeXIV.in.e_deexc
        FeXIV_in[ii,3] = output_FeXIV.in.p_exc
        FeXIV_in[ii,4] = output_FeXIV.in.p_deexc
        FeXIV_in[ii,5] = output_FeXIV.in.ph_exc
        FeXIV_in[ii,6] = output_FeXIV.in.ph_deexc
        FeXIV_in[ii,7] = output_FeXIV.in.rr
        FeXIV_in[ii,8] = output_FeXIV.in.dc
    
    endfor

    out_exp = ["rad_decay","e_exc","e_deexc","p_exc","p_deexc","ph_exc","ph_deexc","ai"]
    in_exp = ["rad_decay","e_exc","e_deexc","p_exc","p_deexc","ph_exc","ph_deexc","rr","dc"]

    save,filename="../sav/CHIANTI/FeX_FeXIV_pop_process.sav",FeX_in,FeX_out,FeXIV_in,FeXIV_out, $ 
            out_exp,in_exp,r_distance,streamer_dens,streamer_temp

end