def set_device():
    import os
    from subprocess import run
    
    cmd = 'nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits'
    cmd += ' | sort -t, -nk2 -r | awk -F , \'FNR == 1 {print $1}\''
    output = run(cmd, shell=True, capture_output=True)
    device = int(output.stdout)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)